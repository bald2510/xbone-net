import os
import random
from PIL import Image

def create_image_grid(image_folders, grid_size=(7, 7), img_size=(256, 256), output_path='dataset_grid.jpg'):
    image_paths = []
    for folder in image_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
    
    if len(image_paths) < grid_size[0] * grid_size[1]:
        raise ValueError(f"Not enough images. Found {len(image_paths)}, need {grid_size[0] * grid_size[1]}")
        
    # Pick randomly
    selected_images = random.sample(image_paths, grid_size[0] * grid_size[1])
    
    # Create blank canvas
    grid_img = Image.new('RGB', (grid_size[1] * img_size[0], grid_size[0] * img_size[1]))
    
    print(f"Generating a {grid_size[0]}x{grid_size[1]} grid for {output_path}...")
    for index, path in enumerate(selected_images):
        row = index // grid_size[1]
        col = index % grid_size[1]
        
        try:
            img = Image.open(path).convert('RGB')
            # Center crop to square to avoid stretching
            width, height = img.size
            min_dim = min(width, height)
            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2
            
            img = img.crop((left, top, right, bottom))
            # Resize
            img = img.resize(img_size, Image.Resampling.LANCZOS)
            
            grid_img.paste(img, (col * img_size[0], row * img_size[1]))
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            
    grid_img.save(output_path)
    print(f"Grid saved successfully to {output_path}")

if __name__ == '__main__':
    # Determine absolute paths based on the current script location to ensure it works anywhere
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    btxrd_folder = os.path.join(base_dir, 'data', 'BTXRD', 'images')
    ctch_folder = os.path.join(base_dir, 'data', 'CTCH', 'images')
    
    # Output inside outputs folder or root
    output_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate for BTXRD
    btxrd_output_path = os.path.join(output_dir, 'btxrd_intro_grid.jpg')
    create_image_grid([btxrd_folder], output_path=btxrd_output_path)
    
    # Generate for CTCH
    ctch_output_path = os.path.join(output_dir, 'ctch_intro_grid.jpg')
    create_image_grid([ctch_folder], output_path=ctch_output_path)
