import torch


def build_optimizer(args, model):
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    if args.sep_optim is None:
        optimizer = getattr(torch.optim, args.optim)(
            [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
            {'params': ed_params, 'lr': args.lr_ed}],
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
        return optimizer
    #following is new code
    optimizer = getattr(torch.optim, args.optim)(
            [{'params': ed_params, 'lr': args.lr_ed}],
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    optimizer_ve_optional = getattr(torch.optim, args.sep_optim)(
            [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve}]
        )
    return optimizer, optimizer_ve_optional


def build_lr_scheduler(args, optimizer, **kwargs):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler
