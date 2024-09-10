import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import datetime
from main import MultiModalVAE  # Import the multimodal VAE class
from torchvision.utils import save_image  # Import save_image


# Set the run name directly in the script
run_name = "multimodal_kl_annealing"


# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIMS = [784, 784]  # Input dimensions for MNIST and Fashion-MNIST
H_DIM = 300
Z_DIM = 40
NUM_EPOCHS = 10  # Increase the number of epochs
BATCH_SIZE = 128
LR_RATE = 3e-4
KL_START_WEIGHT = 0.00001
KL_END_WEIGHT = 0.007
KL_ANNEALING_EPOCHS = 20  # Number of epochs over which to anneal the KL weight


# Data loading
transform = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor()])
mnist = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
fashion_mnist = datasets.FashionMNIST(root="dataset/", train=True, transform=transform, download=True)




class MultimodalMNISTFashionMNIST(Dataset):
   def __init__(self, mnist, fashion_mnist):
       self.mnist = mnist
       self.fashion_mnist = fashion_mnist
       assert len(self.mnist) == len(self.fashion_mnist), "Datasets must be the same size"


   def __len__(self):
       return len(self.mnist)


   def __getitem__(self, idx):
       mnist_img, mnist_label = self.mnist[idx]
       fashion_img, fashion_label = self.fashion_mnist[idx]
       return (mnist_img, fashion_img), (mnist_label, fashion_label)




# Create multimodal dataset
multimodal_dataset = MultimodalMNISTFashionMNIST(mnist, fashion_mnist)
train_loader = DataLoader(multimodal_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Model setup
model = MultiModalVAE(INPUT_DIMS, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="mean")  # Use mean reduction


# TensorBoard writer
log_dir = f"logs/vae/{run_name}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir)


# Training
for epoch in range(NUM_EPOCHS):
   kl_weight = KL_START_WEIGHT + (min(epoch, KL_ANNEALING_EPOCHS) / KL_ANNEALING_EPOCHS) * ( KL_END_WEIGHT - KL_START_WEIGHT)
   loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
   for i, ((mnist_img, fashion_img), _) in loop:
       # Forward pass
       mnist_img = mnist_img.to(DEVICE)
       fashion_img = fashion_img.to(DEVICE)
       images = [mnist_img, fashion_img]
       recons, mu_list, log_var_list, _ = model(images)


       # Compute loss for both modalities
       mnist_reconstructed_loss = loss_fn(recons[0], mnist_img)
       fashion_reconstructed_loss = loss_fn(recons[1], fashion_img)


       # Separate KL divergence calculations for each modality
       mnist_kl_div = torch.mean(
           -0.5 * torch.sum(1 + log_var_list[0] - mu_list[0].pow(2) - log_var_list[0].exp(), dim=1))
       fashion_kl_div = torch.mean(
           -0.5 * torch.sum(1 + log_var_list[1] - mu_list[1].pow(2) - log_var_list[1].exp(), dim=1))


       # Calculate cross-generation loss
       cross_gen_loss_mnist_to_fashion = loss_fn(recons[1], mnist_img)
       cross_gen_loss_fashion_to_mnist = loss_fn(recons[0], fashion_img)


       # Total loss with separate KL annealing and cross-generation loss
       loss = (mnist_reconstructed_loss + fashion_reconstructed_loss) / 2 + (
               mnist_kl_div + fashion_kl_div) / 2 * kl_weight + (
                      cross_gen_loss_mnist_to_fashion + cross_gen_loss_fashion_to_mnist) / 2


       # Backpropagation
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()


       # Update progress bar with losses
       loop.set_postfix(
           epoch=epoch,
           mnist_reconstructed_loss=mnist_reconstructed_loss.item(),
           fashion_reconstructed_loss=fashion_reconstructed_loss.item(),
           mnist_kl_div=mnist_kl_div.item(),
           fashion_kl_div=fashion_kl_div.item(),
           cross_gen_loss_mnist_to_fashion=cross_gen_loss_mnist_to_fashion.item(),
           cross_gen_loss_fashion_to_mnist=cross_gen_loss_fashion_to_mnist.item()
       )


       # Log losses to TensorBoard
       writer.add_scalar("Loss/MNIST_Reconstructed", mnist_reconstructed_loss.item(), epoch * len(train_loader) + i)
       writer.add_scalar("Loss/FashionMNIST_Reconstructed", fashion_reconstructed_loss.item(),
                         epoch * len(train_loader) + i)
       writer.add_scalar("Loss/MNIST_KL_divergence", mnist_kl_div.item(), epoch * len(train_loader) + i)
       writer.add_scalar("Loss/FashionMNIST_KL_divergence", fashion_kl_div.item(), epoch * len(train_loader) + i)
       writer.add_scalar("Loss/CrossGen_MNIST_to_Fashion", cross_gen_loss_mnist_to_fashion.item(),
                         epoch * len(train_loader) + i)
       writer.add_scalar("Loss/CrossGen_Fashion_to_MNIST", cross_gen_loss_fashion_to_mnist.item(),
                         epoch * len(train_loader) + i)
       writer.add_scalar("KL_weight", kl_weight, epoch * len(train_loader) + i)


# Close the SummaryWriter
writer.close()




# Model inference function
def inference(model, image, source_modality_idx, target_modality_idx=None, device=DEVICE):
   """
   Generate cross-modal or same-modal output given an input image and target modality index.


   Args:
   - model (nn.Module): Trained multimodal VAE model.
   - image (Tensor): Input image tensor.
   - source_modality_idx (int): Index of the source modality.
   - target_modality_idx (int or None): Index of the target modality to generate.
     If None, generates image in the same modality as source.
   - device (torch.device): Device to perform inference on.


   Returns:
   - generated_image (Tensor): Generated image tensor in the target or same modality.
   """
   model.eval()
   with torch.no_grad():
       image = image.to(device)
       # If target_modality_idx is None, use source_modality_idx for reconstruction in the same modality
       if target_modality_idx is None:
           target_modality_idx = source_modality_idx
       # Create a list with None for other modalities except the source
       images = [None] * len(model.encoders_conv)
       images[source_modality_idx] = image
       mu_list, log_var_list = model.encode(images)
       z = model.reparameterize(mu_list[source_modality_idx], log_var_list[source_modality_idx])
       generated_images = model.decode(z)
       generated_image = generated_images[target_modality_idx]
   return generated_image






# Example of inference after training
test_images, _ = next(iter(train_loader))  # Get a batch of test images
mnist_img, fashion_img = test_images[0], test_images[1]


# Save generated images for evaluation
# Cross-generated images
generated_fashion_img = inference(model, mnist_img[0].unsqueeze(0), source_modality_idx=0, target_modality_idx=1)
save_image(generated_fashion_img, 'generated_fashion_mnist_from_mnist.png')


generated_mnist_img = inference(model, fashion_img[0].unsqueeze(0), source_modality_idx=1, target_modality_idx=0)
save_image(generated_mnist_img, 'generated_mnist_from_fashion_mnist.png')


# Non-cross-generated images (reconstruction in the same modality)
reconstructed_mnist_img = inference(model, mnist_img[0].unsqueeze(0), source_modality_idx=0, target_modality_idx=None)
save_image(reconstructed_mnist_img, 'reconstructed_mnist.png')


reconstructed_fashion_img = inference(model, fashion_img[0].unsqueeze(0), source_modality_idx=1, target_modality_idx=None)
save_image(reconstructed_fashion_img, 'reconstructed_fashion_mnist.png')


# Generate more images for each digit
for idx in range(10):
   # Cross-generated




   # Non-cross-generated (reconstruction in the same modality)
   reconstructed_mnist_img = inference(model, mnist_img[idx].unsqueeze(0), source_modality_idx=0, target_modality_idx=None)
   save_image(reconstructed_mnist_img, f'reconstructed_mnist_{idx}.png')


   reconstructed_fashion_img = inference(model, fashion_img[idx].unsqueeze(0), source_modality_idx=1, target_modality_idx=None)
   save_image(reconstructed_fashion_img, f'reconstructed_fashion_mnist_{idx}.png')