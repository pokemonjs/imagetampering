from multiprocessing import freeze_support

from torch.utils.data import DataLoader

from dataset.semi import SemiDataset
from main import label
from main import parse_args, init_basic_elems

import torch
print("cuda", torch.cuda.is_available())

# args = parse_args()
# if args.epochs is None:
#     args.epochs = 20  # {'pascal': 80, 'cityscapes': 240, 'fake':80}[args.dataset]
# if args.lr is None:
#     args.lr = 0.001  # {'pascal': 0.001, 'cityscapes': 0.004, 'fake':0.001}[args.dataset] / 16 * args.batch_size
# if args.crop_size is None:
#     args.crop_size = 256  # {'pascal': 321, 'cityscapes': 721, 'fake':321}[args.dataset]
# dataset = SemiDataset(args.dataset, args.data_root, "train", args.crop_size, args.labeled_id_path)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
# model, _ = init_basic_elems(args)
# next(iter(dataloader))

# if __name__ == '__main__':
    # freeze_support()
    # label(model, dataloader, args)
