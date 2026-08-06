import os
import random
from PIL import Image

def create_image_grid(image_folders, grid_size=(7, 7), img_size=(256, 256), output_path='grid.jpg'):
    image_paths = []
    for folder in image_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
    
    # Shuffle and pick 49 images
    if len(image_paths) < grid_size[0] * grid_size[1]:
        raise ValueError("Not enough images to fill the grid.")
        
    selected_images = random.sample(image_paths, grid_size[0] * grid_size[1])
    
    grid_img = Image.new('RGB', (grid_size[1] * img_size[0], grid_size[0] * img_size[1]))
    
    for index, path in enumerate(selected_images):
        row = index // grid_size[1]
        col = index % grid_size[1]
        
        try:
            img = Image.open(path).convert('RGB')
            # Center crop or simply resize? Let's simply resize first or thumbnail
            # A better approach is to crop to square to avoid distortion
            width, height = img.size
            min_dim = min(width, height)
            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2
            
            img = img.crop((left, top, right, bottom))
            img = img.resize(img_size, Image.Resampling.LANCZOS)
            
            grid_img.paste(img, (col * img_size[0], row * img_size[1]))
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            
    grid_img.save(output_path)
    print(f"Grid saved to {output_path}")

if __name__ == '__main__':
    folders = [
        r'c:\Users\lebat\Documents\Github\xbone-net\data\BTXRD\images',
        r'c:\Users\lebat\Documents\Github\xbone-net\data\CTCH\images'
    ]
    output_path = r'c:\Users\lebat\Documents\Github\xbone-net\tmp\dataset_intro_grid.jpg'
    create_image_grid(folders, output_path=output_path)
