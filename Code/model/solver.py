import json
import logging
from pathlib import Path
import os
import time
import torch
from model import augment, distrib
from model.enhance import enhance
from model.evaluate import evaluate
from model.utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress
from model.loss import mse_loss, stftm_loss, perceptual_loss

logger = logging.getLogger(__name__)


class Solver(object):
    def __init__(self, data, model, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        self.model = model
        self.dmodel = distrib.wrap(model)
        self.optimizer = optimizer

        augments = []
        if args.remix:
            augments.append(augment.Remix())
        if args.bandmask:
            augments.append(augment.BandMask(args.bandmask, sample_rate=args.sample_rate))
        if args.shift:
            augments.append(augment.Shift(args.shift, args.shift_same))
        if args.revecho:
            augments.append(
                augment.RevEcho(args.revecho))
        self.augment = torch.nn.Sequential(*augments)

        self.device = args.device
        self.epochs = args.epochs
        self.mse_loss = mse_loss()
        self.stftm_loss = stftm_loss()
        self.pcep_loss = perceptual_loss()

        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        self.checkpoint = args.checkpoint
        if self.checkpoint:
            self.checkpoint_file = Path(args.checkpoint_file)
            self.best_file = Path(args.best_file)
            logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.history_file = args.history_file

        self.best_state = None
        self.restart = args.restart
        self.history = []
        self.samples_dir = args.samples_dir
        self.num_prints = args.num_prints
        self.args = args
        self._reset()

    def _serialize(self):
        package = {}
        package['model'] = serialize_model(self.model)
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        tmp_path = str(self.checkpoint_file) + ".tmp"
        torch.save(package, tmp_path)
        os.rename(tmp_path, self.checkpoint_file)
        model = package['model']
        model['state'] = self.best_state
        tmp_path = str(self.best_file) + ".tmp"
        torch.save(model, tmp_path)
        os.rename(tmp_path, self.best_file)

    def _reset(self):
        load_from = None
        load_best = False
        keep_history = True
        if self.checkpoint and self.checkpoint_file.exists() and not self.restart:
            load_from = self.checkpoint_file
        elif self.continue_from:
            load_from = self.continue_from
            load_best = self.args.continue_best
            keep_history = False
        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            if load_best:
                self.model.load_state_dict(package['best_state'])
            else:
                self.model.load_state_dict(package['model']['state'])
            if 'optimizer' in package and not load_best:
                self.optimizer.load_state_dict(package['optimizer'])
            if keep_history:
                self.history = package['history']
            self.best_state = package['best_state']


    def train(self):
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")
        for epoch in range(len(self.history), self.epochs):
            self.model.train()
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            train_loss = self._run_one_epoch(epoch)
            logger.info(
                bold(f'Train Summary | End of Epoch {epoch + 1} | '
                     f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}'))
            if self.cv_loader:
                logger.info('-' * 70)
                logger.info('Cross validation...')
                self.model.eval()
                with torch.no_grad():
                    valid_loss = self._run_one_epoch(epoch, cross_valid=True)
                logger.info(
                    bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                         f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}'))
            else:
                valid_loss = 0
            best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
            metrics = {'train': train_loss, 'valid': valid_loss, 'best': best_loss}
            if valid_loss == best_loss:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = copy_state(self.model.state_dict())
            if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                with swap_state(self.model, self.best_state):
                    pesq, stoi = evaluate(self.args, self.model, self.tt_loader)
                metrics.update({'pesq': pesq, 'stoi': stoi})
                logger.info('Enhance and save samples...')
                enhance(self.args, self.model, self.samples_dir)
            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))
            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                if self.checkpoint:
                    self._serialize()
                    logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())

    def _run_one_epoch(self, epoch, cross_valid=False):
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        data_loader.epoch = epoch
        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        for i, data in enumerate(logprog):
            noisy, clean = [x.to(self.device) for x in data]
            if not cross_valid:
                sources = torch.stack([noisy - clean, clean])
                noise, clean = sources
                noisy = noise + clean
            estimate = self.dmodel(noisy)
            if self.args.stft_loss:
                loss_mask = torch.ones(clean.shape).cuda()
                loss_time = self.mse_loss(estimate, clean, loss_mask)
                loss_freq = self.stftm_loss(estimate, clean, loss_mask)
                loss_pcep = self.pcep_loss(estimate, clean, loss_mask)
                loss = 0.8 * loss_time + 0.2 * (loss_freq + loss_pcep)
                if not cross_valid:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()
            logprog.update(loss=format(total_loss / (i + 1), ".5f"))
            del loss, estimate
        return distrib.average([total_loss / (i + 1)], i + 1)[0]
