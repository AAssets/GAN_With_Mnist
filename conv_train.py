import os
import importlib

import torch
import torchvision
import torch.nn as nn

import argparse
import swanlab

import torchvision.utils as vutils

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
                                                torchvision.transforms.Normalize((0.5,), (0.5,)),
                                            ]
                                                                                )
                                        )
    

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    module =importlib.import_module("models."+args.model)


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    generator = module.Generator(args)
    generator.apply(weights_init)
    discriminator = module.Discriminator(args)
    discriminator.apply(weights_init)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.4, 0.8), weight_decay=0.0001)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.4, 0.8), weight_decay=0.0001)

    loss_fn = nn.BCELoss()
    labels_one = 1
    labels_zero = 0


    if use_gpu:
        print("use gpu for training")
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        loss_fn = loss_fn.cuda()

    iteration_num=0



    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Starting Training Loop...")
    for epoch in range(args.epoch):
        for i, data in enumerate(dataloader, 0):
            iteration_num+=1
            # (1) Update the discriminator with real data
            discriminator.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            label = torch.full((args.batch_size,), labels_one, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            real_loss = loss_fn(output, label)
            # Calculate gradients for D in backward pass
            real_loss.backward()
            D_x = output.mean().item()

            # (2) Update the discriminator with fake data
            # Generate batch of latent vectors
            z = torch.randn(args.batch_size, args.latent_size, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(z)
            label.fill_(labels_zero)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            fake_loss = loss_fn(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            fake_loss.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            d_loss = real_loss + fake_loss
            # Update D
            d_optimizer.step()

            # (3) Update the generator with fake data
            generator.zero_grad()
            label.fill_(labels_one)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            g_loss = loss_fn(output, label)
            # Calculate gradients for G
            g_loss.backward()
            D_G_z2 = output.mean().item()
            # Update G
            g_optimizer.step()

            recons_loss = torch.abs(fake-real_cpu).mean()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, args.epoch, i, len(dataloader),
                        d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iteration_num % 500 == 0) or ((epoch == args.epoch-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(z).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            if i % 50 == 0:
                print(f"step:{len(dataloader)*epoch+i}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")

        if epoch % 1 == 0:
            image = fake[:16].data
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
    parser.add_argument('--latent_size', type=int, default='100', help='latent size')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--image_size', type=int, default=64, help='image size')
    parser.add_argument('--model', type=str, default="conv_model", help='use model name')
    args = parser.parse_args()

    return args

if __name__=='__main__':
    args=get_opts()
    main(args)
