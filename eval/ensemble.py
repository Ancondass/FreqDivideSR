
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from config import Config
from sr_model import create_model


def load_multiple_checkpoints(checkpoint_paths, device):
    models = []
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f'Loading checkpoint from {path}...')
            model = create_model()
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            models.append(model)
            val_psnr = checkpoint.get('val_psnr', 0.0)
            print(f"  Loaded model with Val PSNR={val_psnr:.2f} dB")
        else:
            print(f'Warning: Checkpoint not found at {path}')
    return models


def ensemble_predict(models, lr_img, device, scale=4):
    sr_outputs = []
    
    with torch.no_grad():
        for model in models:
            sr = model(lr_img, scale=scale)
            sr_outputs.append(sr)
    
    sr_stack = torch.stack(sr_outputs, dim=0)
    sr_ensemble = torch.mean(sr_stack, dim=0)
    
    return sr_ensemble


def predict_with_ensemble(models, lr_tensor, device, scale=4, batch_size=1):
    sr_outputs = []
    
    with torch.no_grad():
        for i in range(0, lr_tensor.size(0), batch_size):
            batch = lr_tensor[i:i+batch_size].to(device)
            sr_batch = ensemble_predict(models, batch, device, scale=scale)
            sr_outputs.append(sr_batch.cpu())
    
    return torch.cat(sr_outputs, dim=0)


def inference_single_image_ensemble(image_path, checkpoint_paths, output_path, device, scale=4):
    print(f'\n=== Ensemble Inference ===')
    print(f'Input: {image_path}')
    print(f'Output: {output_path}')
    print(f'Checkpoints: {checkpoint_paths}')
    
    models = load_multiple_checkpoints(checkpoint_paths, device)
    
    if not models:
        print('Error: No models loaded!')
        return
    
    print(f'\nLoaded {len(models)} models for ensemble')
    
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    print(f'Original image size: {original_size}')
    
    hr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    lr_transform = transforms.Compose([
        transforms.Resize((original_size[1] // scale, original_size[0] // scale), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    lr_img = lr_transform(img).unsqueeze(0)
    
    print(f'LR size: {lr_img.shape[2]}x{lr_img.shape[3]}')
    
    sr_img = predict_with_ensemble(models, lr_img, device, scale=scale, batch_size=1)
    
    sr_img = sr_img.squeeze(0).cpu()
    sr_img = sr_img * 0.5 + 0.5
    sr_img = torch.clamp(sr_img, 0, 1)
    
    sr_pil = transforms.ToPILImage()(sr_img)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sr_pil.save(output_path)
    
    print(f'Saved SR image to {output_path}')
    print(f'SR image size: {sr_pil.size}')
    
    upscale_info = f'{original_size[0]}x{original_size[1]} -> {sr_pil.size[0]}x{sr_pil.size[1]}'
    print(f'Upscale: {upscale_info}')
    
    return sr_pil


def inference_folder_ensemble(input_folder, output_folder, checkpoint_paths, device, scale=4):
    print(f'\n=== Ensemble Folder Inference ===')
    print(f'Input folder: {input_folder}')
    print(f'Output folder: {output_folder}')
    print(f'Scale: x{scale}')
    
    models = load_multiple_checkpoints(checkpoint_paths, device)
    
    if not models:
        print('Error: No models loaded!')
        return
    
    print(f'\nLoaded {len(models)} models for ensemble')
    
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]
    image_files.sort()
    
    print(f'Found {len(image_files)} images')
    
    for img_file in tqdm(image_files, desc='Processing'):
        img_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, f'sr_{img_file}')
        
        try:
            img = Image.open(img_path).convert('RGB')
            original_size = img.size
            
            hr_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            lr_transform = transforms.Compose([
                transforms.Resize((original_size[1] // scale, original_size[0] // scale), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            lr_img = lr_transform(img).unsqueeze(0)
            sr_img = predict_with_ensemble(models, lr_img, device, scale=scale, batch_size=1)
            sr_img = sr_img.squeeze(0).cpu()
            sr_img = sr_img * 0.5 + 0.5
            sr_img = torch.clamp(sr_img, 0, 1)
            
            sr_pil = transforms.ToPILImage()(sr_img)
            sr_pil.save(output_path)
            
        except Exception as e:
            print(f'Error processing {img_file}: {e}')
    
    print(f'\nCompleted! Output saved to {output_folder}')


def main():
    device = Config.DEVICE
    print(f'Using device: {device}')
    
    checkpoint_paths = [
        'checkpoints/best_model.pth',
        'checkpoints/final_model.pth',
    ]
    
    image_path = 'input.jpg'
    output_path = 'output_ensemble.jpg'
    
    if os.path.exists(image_path):
        inference_single_image_ensemble(image_path, checkpoint_paths, output_path, device)
    else:
        print(f'Image not found at {image_path}, using folder inference instead')
        input_folder = 'inputs'
        output_folder = 'outputs_ensemble'
        if os.path.exists(input_folder):
            inference_folder_ensemble(input_folder, output_folder, checkpoint_paths, device)
        else:
            print('No input folder found. Please place images in "inputs" folder or specify image_path')


if __name__ == '__main__':
    main()
