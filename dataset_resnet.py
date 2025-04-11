import logging
import os
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fashion_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FashionDataset')


class FashionDataset(Dataset):

    def __init__(self, metadata_path, cropped_images_dir, heatmaps_dir, transform=None,
                 use_cache=True, cache_size=100, validate_files=True):

        super().__init__()
        logger.info(f"Initializing FashionDataset with metadata: {metadata_path}")
        print((f"Load metadata from: {metadata_path}"))
   
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found: {metadata_path}")
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        if not os.path.exists(cropped_images_dir):
            logger.error(f"Cropped images directory not found: {cropped_images_dir}")
            raise FileNotFoundError(f"Cropped images directory not found: {cropped_images_dir}")

        if not os.path.exists(heatmaps_dir):
            logger.error(f"Heatmaps directory not found: {heatmaps_dir}")
            raise FileNotFoundError(f"Heatmaps directory not found: {heatmaps_dir}")

        try:
            self.metadata = pd.read_csv(metadata_path)
            logger.info(f"Loaded metadata with {len(self.metadata)} entries")
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            raise

        self.cropped_images_dir = cropped_images_dir
        self.heatmaps_dir = heatmaps_dir

        self.transform = transform or self.default_transforms()
        logger.info(f"Using transforms: {self.transform}")

       
        self.attribute_columns = self.metadata.columns[4:-2].tolist()   # extract attribute columns
        logger.info(f"Extracted {len(self.attribute_columns)} attribute columns")

       
        original_categories = sorted(self.metadata['category_label'].unique())  #  remapping categ lable
        logger.info(f"Original unique categories: {original_categories}")

  
        valid_categories = [c for c in range(1, 51) if c not in [38, 45, 49, 50]]         # removing missin
        logger.info(f"Valid categories after filtering: {valid_categories}")

        self.metadata['original_category_label'] = self.metadata['category_label'].copy()

        self.category_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(valid_categories))}

        logger.info(f"Category mapping: {self.category_mapping}")

  
        self.metadata = self.metadata[self.metadata['category_label'].isin(valid_categories)]  
        self.metadata['category_label'] = self.metadata['category_label'].map(self.category_mapping)

        assert self.metadata['category_label'].min() == 0, "Error: Category labels should start from 0"
        assert self.metadata['category_label'].max() == 45, "Error: Category labels should range from 0 to 45"


        unique_labels = sorted(self.metadata['category_label'].unique().tolist())
        print("Final Unique Categories in Dataset:", unique_labels)
        assert 45 in unique_labels, "Error: Category 45 is missing from the dataset!"


        
        min_label = self.metadata['category_label'].min()  # verify the labels are now 0-indexed
        max_label = self.metadata['category_label'].max()
        logger.info(f"After remapping: category_label min={min_label}, max={max_label}")

        self.metadata['category_type'] = self.metadata['category_type'] - 1  # 1-3 to 0-2


        logger.info(f"Final category labels after remapping: {sorted(self.metadata['category_label'].unique())}")
        print(f"Final Category Labels: {sorted(self.metadata['category_label'].unique())}")  # Should show 0 to 45

       
        self.category_labels = self.metadata['category_label']
        self.category_types = self.metadata['category_type']

    
        try:
            self.category_names = dict(zip(
                self.metadata['category_label'].unique(),
                self.metadata['category_name'].unique() if 'category_name' in self.metadata.columns
                else self.metadata['category_label'].unique()
            ))
            logger.info(f"Created category mapping with {len(self.category_names)} categories")
        except Exception as e:
            logger.warning(f"Could not create category name mapping: {str(e)}")
            self.category_names = dict(zip(self.metadata['category_label'].unique(),
                                        self.metadata['category_label'].unique()))

     
        self.cache_size = cache_size  # store cache_size for later use
        self.use_cache = use_cache
        if use_cache:
            logger.info(f"Enabling LRU cache with size {cache_size}")
            self.load_image = lru_cache(maxsize=cache_size)(self._load_image)
            self.load_heatmap = lru_cache(maxsize=cache_size)(self._load_heatmap)
        else:
            self.load_image = self._load_image
            self.load_heatmap = self._load_heatmap


     
        if validate_files:
            self.validate_dataset_files()

    def validate_dataset_files(self):
      
        logger.info("Validating dataset files...")

        missing_images = 0
        missing_heatmaps = 0

        sample_size = min(100, len(self.metadata))
        indices = random.sample(range(len(self.metadata)), sample_size)

        for idx in indices:
            row = self.metadata.iloc[idx]


            img_path = os.path.join(self.cropped_images_dir, row['image_name'])


            if not os.path.exists(img_path):
                missing_images += 1
                logger.warning(f"Image not found: {img_path}")

            heatmap_path = os.path.join(self.heatmaps_dir, f"{os.path.splitext(row['image_name'])[0]}.npy")
            if not os.path.exists(heatmap_path):
                missing_heatmaps += 1
                logger.warning(f"Heatmap not found: {heatmap_path}")

        if missing_images > 0 or missing_heatmaps > 0:
            logger.warning(f"Found {missing_images} missing images and {missing_heatmaps} missing heatmaps in sample of {sample_size}")
        else:
            logger.info(f"Validated {sample_size} random samples, all files exist")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        try:
         
            row = self.metadata.iloc[idx]

            image = self.load_image(row['image_name'])

            heatmap = self.load_heatmap(row['image_name'])

            attributes = torch.tensor(row[self.attribute_columns].values.astype(np.float32))

            category_label = torch.tensor(int(row['category_label']), dtype=torch.long)
            assert category_label >= 0, f"Error: Found category -1 in dataset at index {idx}!"
            category_type = torch.tensor(row['category_type'], dtype=torch.long)

            # Apply transforms
            if self.transform:
                seed = np.random.randint(2147483647)

                
                if not isinstance(image, torch.Tensor):
                    image = transforms.ToTensor()(image)  #  Convert PIL image to tensor first


               
                if not any(isinstance(t, transforms.ToTensor) for t in self.transform.transforms):
                    image = self.transform(image)

               
                random.seed(seed)
                torch.manual_seed(seed)

              
                random.seed(seed)
                torch.manual_seed(seed)
                heatmap = self.transform_heatmap(heatmap)

            return {
                'image': image,
                'heatmap': heatmap,
                'attributes': attributes,
                'category_label': category_label,
                'category_type': category_type,
                'image_name': row['image_name']  # For debugging/visualization
            }



        except Exception as e:
            logger.error(f"Error in __getitem__: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"Error processing item {idx} ({row['image_name'] if 'row' in locals() else 'unknown'}): {str(e)}")
         
            if idx > 0:  
                return self.__getitem__(0)
            else:
                raise

    def __getstate__(self):
        state = self.__dict__.copy()
        
        if 'load_image' in state:
            del state['load_image']
        if 'load_heatmap' in state:
            del state['load_heatmap']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
       
        if self.use_cache:
            self.load_image = lru_cache(maxsize=self.cache_size)(self._load_image)
            self.load_heatmap = lru_cache(maxsize=self.cache_size)(self._load_heatmap)
        else:
            self.load_image = self._load_image
            self.load_heatmap = self._load_heatmap


    def _load_image(self, image_name):
        """Helper function to load image with error handling"""
        img_path = os.path.join(self.cropped_images_dir, image_name)
        try:
            if not os.path.exists(img_path):
                logger.error(f"Image not found: {img_path}")
                raise FileNotFoundError(f"Image not found: {img_path}")

            image = Image.open(img_path).convert('RGB')
           
            if image.size != (224, 224):
                logger.warning(f"Image {img_path} has unexpected dimensions {image.size}, resizing to (224, 224)")
                image = image.resize((224, 224))

            return image

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            
            return Image.new('RGB', (224, 224), color='gray')

    def _load_heatmap(self, image_name):
        """Helper function to load heatmap with error handling"""
        heatmap_path = os.path.join(self.heatmaps_dir, f"{os.path.splitext(image_name)[0]}.npy")
        try:
            if not os.path.exists(heatmap_path):
                logger.error(f"Heatmap not found: {heatmap_path}")
                raise FileNotFoundError(f"Heatmap not found: {heatmap_path}")

            heatmap = np.load(heatmap_path)
           
            if heatmap.shape != (224, 224):
                logger.warning(f"Heatmap {heatmap_path} has unexpected dimensions {heatmap.shape}, resizing to (224, 224)")
                
                heatmap = np.array(Image.fromarray(heatmap).resize((224, 224), Image.BILINEAR))

            return heatmap

        except Exception as e:
            logger.error(f"Error loading heatmap {heatmap_path}: {str(e)}")
            # Return a blank heatma
            return np.zeros((224, 224), dtype=np.float32)

    def default_transforms(self):
        """Default transforms for images and heatmaps"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                                 std=[0.229, 0.224, 0.225])
        ])

    def transform_heatmap(self, heatmap):
        """Special transforms for heatmaps"""
       
        heatmap = torch.from_numpy(heatmap).float().unsqueeze(0)  # [1, H, W]

        # Apply same transformations 
        if self.transform:
            
            if isinstance(self.transform, transforms.Compose):
                for t in self.transform.transforms:
                    if isinstance(t, (transforms.RandomHorizontalFlip,
                                      transforms.RandomRotation,
                                      transforms.RandomAffine,
                                      transforms.RandomResizedCrop)):
                        heatmap = t(heatmap)
        return heatmap

    def get_category_mapping(self):
        """Get mapping of category labels to human-readable names"""
        return self.category_names

    def get_attribute_names(self):
        """Get list of attribute names in order"""
        return self.attribute_columns

    def get_stats(self):
        """Get dataset statistics"""
        logger.info("Calculating dataset statistics...")

        stats = {
            'num_samples': len(self.metadata),
            'num_attributes': len(self.attribute_columns),
            'num_categories': len(self.category_labels.unique()),
            'attribute_distributions': {attr: self.metadata[attr].value_counts().to_dict()
                                        for attr in self.attribute_columns[:5]}, 
            'category_distribution': self.metadata['category_label'].value_counts().to_dict(),
            'type_distribution': self.metadata['category_type'].value_counts().to_dict()
        }

        return stats