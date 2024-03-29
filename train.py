import os
import importlib

import torch
import torchvision
import torch.nn as nn

import argparse
import swanlab


from datetime import datetime



def main(args):

    # 启动日志
    logdir="./logs"
    swanlab.init(
    experiment_name=args.exp,
    config=args,
    logdir=logdir
        )

    use_gpu = torch.cuda.is_available()

    # 生成文件夹
    # 获取当前时间
    current_time = datetime.now()
    # 格式化时间为字符串，例如：2024-02-23_15-30-45
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    OUTPUT_PATH=f'./outputs/{args.model}_{timestamp}/'
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Data setting
    dataset = torchvision.datasets.MNIST("./data/mnist_data", train=True, download=True,
                                        transform=torchvision.transforms.Compose(
                                            [
                                                torchvision.transforms.Resize(args.image_size),
                                                torchvision.transforms.ToTensor(),
                                            ]
                                                                                )
                                        )
    

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    module =importlib.import_module("models."+args.model)

    generator = module.Generator(args)
    discriminator = module.Discriminator(args)


    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.4, 0.8), weight_decay=0.0001)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.4, 0.8), weight_decay=0.0001)

    loss_fn = nn.BCELoss()
    labels_one = torch.ones(args.batch_size,1)
    labels_zero = torch.zeros(args.batch_size,1)


    if use_gpu:
        print("use gpu for training")
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        loss_fn = loss_fn.cuda()
        labels_one = labels_one.to("cuda")
        labels_zero = labels_zero.to("cuda")

    # Training
    iteration_num=0
    for epoch in range(args.epoch):
        for i, mini_batch in enumerate(dataloader):
            iteration_num+=1
            gt_images, _ = mini_batch


            z = torch.randn(args.batch_size, args.latent_size)

            if use_gpu:
                gt_images = gt_images.to("cuda")
                z = z.to("cuda")

            pred_images = generator(z)
            g_optimizer.zero_grad()

            recons_loss = torch.abs(pred_images-gt_images).mean()

            g_loss = recons_loss*0.05 + loss_fn(discriminator(pred_images), labels_one)

            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()

            real_loss = loss_fn(discriminator(gt_images), labels_one)
            fake_loss = loss_fn(discriminator(pred_images.detach()), labels_zero)
            d_loss = (real_loss + fake_loss)


            d_loss.backward()
            d_optimizer.step()

            if i % 50 == 0:
                print(f"step:{len(dataloader)*epoch+i}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")

        if epoch % 10 == 0:
            image = pred_images[:16].data
            torchvision.utils.save_image(image,os.path.join(OUTPUT_PATH, f'image_{len(dataloader)*epoch+i}.png'), nrow=4)

        swanlab.log({"g loss": g_loss / 100},iteration_num)
        swanlab.log({"d loss": d_loss / 100},iteration_num)
        swanlab.log({"real loss": real_loss / 100},iteration_num)
        swanlab.log({"fake loss": fake_loss / 100},iteration_num)


def get_opts():
    parser = argparse.ArgumentParser(description='The parser for text generation')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate of train')
    parser.add_argument('--epoch', type=int, default=200, help='epoch of train')
    parser.add_argument('--exp', type=str, default='GAN', help='exp name')
    parser.add_argument('--latent_size', type=int, default='64', help='latent size')
    parser.add_argument('--image_size', type=int, default=28, help='image size')
    parser.add_argument('--model', type=str, default="base", help='use model name')
    args = parser.parse_args()

    return args

if __name__=='__main__':
    args=get_opts()
    main(args)
