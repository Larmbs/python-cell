import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import argparse
import pathlib

from model import CAModel
 

def load_image(path, size=40):
    """Load an image
    
    Parameters
    ----------
    path : String
        Path to image file RGBA
        
    size : int
        The image will be resized to a square with side length of 'size'
        
    Returns 
    -------
    torch.Tensor
        4D float image of shape '(1, 4, size, size)'. The RGB channels
        are premultiplied by the alpha channel
    """
    img = Image.open(path)
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    img = np.array(img, dtype=np.float32) / 255.0
    img[..., :3] *= img[..., 3:]
    
    return torch.from_numpy(img).permute(2, 0, 1)[None, ...]

def to_rgb(img_rgba):
    """Convert RGBA image to RGB"""
    rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0, 1)

def make_seed(size, n_channels):
    """Create starting image"""
    x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    x[:, 3:, size//2, size//2] = 1
    return x

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Training script for the Cellular Automata"
    )
    parser.add_argument("img", type=str, help="Path to the image you want to reproduce")
    parser.add_argument(
        "-b", 
        "--batch-size", 
        type=int, 
        default=8, 
        help="Batch size. Sample will be taken randomly from pool"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device to use",
        choices=("cpu", "cuda"),
    )
    parser.add_argument(
        "-e",
        "--eval-frequency",
        type=int,
        default=600,
        help="Evaluation frequency.",
    )
    parser.add_argument(
        "-i",
        "--eval-iterations",
        type=int,
        default=300,
        help="Number of iterations when evaluating.",
    )
    parser.add_argument(
        "-n",
        "--n-batches",
        type=int,
        default=5000,
        help="Number of iterations when evaluating.",
    )
    parser.add_argument(
        "-c",
        "--n-channels",
        type=int,
        default=16,
        help="Number of channels of the input tensor.",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="Folder where you put all your logs.",
    )
    parser.add_argument(
        "-p",
        "--padding",
        type=int,
        default=16,
        help="Add padding to image.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=1024,
        help="Size of training pool.",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=40,
        help="Image size.",
    )
    
    args = parser.parse_args()
    print(vars(args))

    device = torch.device(args.device)
    
    log_path = pathlib.Path(args.logdir)
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)
    
    target_img_ = load_image(args.img, size=args.size)
    p = args.padding
    target_img_ = nn.functional.pad(target_img_, (p, p, p, p), "constant", 0)
    target_img_ = target_img_.to(device)
    target_img_ = target_img_.repeat(args.batch_size, 1, 1, 1)
    
    writer.add_image("ground truth", to_rgb(target_img_)[0])
    
    model = CAModel(n_channels=args.n_channels, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    
    seed = make_seed(args.size, args.n_channels).to(device)
    seed = nn.functional.pad(seed, (p, p, p, p), "constant", 0)
    pool = seed.clone().repeat(args.pool_size, 1, 1, 1)
    
    for it in tqdm(range(args.n_batches)):
        batch_ixs = np.random.choice(
            args.pool_size, args.batch_size, replace=False
        ).tolist()
        
        x = pool[batch_ixs]
        for i in range(np.random.randint(64, 96)):
            x = model(x)
            
        loss_batch = ((target_img_ - x[:, :4, ...]) ** 2).mean(dim=[1, 2, 3])
        loss = loss_batch.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("train/loss", loss, it)
        
        argmax_batch = loss_batch.argmax().item()
        argmax_pool = batch_ixs[argmax_batch]
        remaining_batch = [i for i in range(args.batch_size) if i != argmax_batch]
        remaining_pool = [i for i in batch_ixs if i != argmax_pool]
        
        pool[argmax_pool] = seed.clone()
        pool[remaining_pool] = x[remaining_batch].detach()
        
        if it % args.eval_frequency == 0:
            x_eval = seed.clone()
            
            eval_video = torch.empty(1, args.eval_iterations, 3, *x_eval.shape[2:])
            
            for it_eval in range(args.eval_iterations):
                x_eval = model(x_eval)
                x_eval_out = to_rgb(x_eval[:, :4].detach().cpu())
                eval_video[0, it_eval] = x_eval_out
                
            writer.add_video("eval", eval_video, it, fps=60)

if __name__ == "__main__":
    main()