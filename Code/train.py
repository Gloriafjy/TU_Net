import os
import logging
import hydra

logger = logging.getLogger(__name__)


def run(args):
    import torch

    from model import distrib
    from model.data import NoisyCleanSet
    from model.tunet import TUNET
    from model.solver import Solver
    distrib.init(args)

    model = TUNET(**args.tunet)

    if args.show:
        logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
        logger.info('Size: %.1f MB', mb)
        if hasattr(model, 'valid_length'):
            field = model.valid_length(1)
            logger.info('Field: %.1f ms', field / args.sample_rate * 1000)
        return

    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    length = int(args.segment * args.sample_rate)
    stride = int(args.stride * args.sample_rate)
    if hasattr(model, 'valid_length'):
        length = model.valid_length(length)
    kwargs = {"matching": args.dset.matching, "sample_rate": args.sample_rate}
    tr_dataset = NoisyCleanSet(
        args.dset.train, length=length, stride=stride, pad=args.pad, **kwargs)
    tr_loader = distrib.loader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.dset.valid:
        cv_dataset = NoisyCleanSet(args.dset.valid, **kwargs)
        cv_loader = distrib.loader(cv_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        cv_loader = None
    if args.dset.test:
        tt_dataset = NoisyCleanSet(args.dset.test, **kwargs)
        tt_loader = distrib.loader(tt_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        tt_loader = None
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        model.cuda()

    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, args.beta2))
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    solver = Solver(data, model, optimizer, args)
    solver.train()


def _main(args):
    global __file__
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("tunet").setLevel(logging.DEBUG)
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    run(args)


@hydra.main(config_path="conf/config.yaml")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        os._exit(1)


if __name__ == "__main__":
    main()
