#dataset vit
import logging
import os
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Set up logging
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
    def __init__(self, metadata_path, cropped_images_dir, heatmaps_dir, 
                 attribute_groups, transform=None, use_cache=True, 
                 cache_size=100, validate_files=True):
        
        super().__init__()
        logger.info(f"Initializing FashionDataset with metadata: {metadata_path}")
      
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

            # We keep these lines for reference but won't rely on them for actual file paths:
            self.metadata['cropped_image_path'] = self.metadata['cropped_image_path'].str.replace('\\', '/', regex=False)
            self.metadata['heatmaps_path'] = self.metadata['heatmaps_path'].str.replace('\\', '/', regex=False)
            self.metadata['heatmaps_path'] = self.metadata['heatmaps_path'].str.replace('.npy', '.npz', regex=False)

        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            raise

       
        self.cropped_images_dir = cropped_images_dir
        self.heatmaps_dir = heatmaps_dir
        self.transform = transform or self.default_transforms()
        self.attribute_groups = attribute_groups
        self.group_to_indices = self._create_group_to_indices()
       
        self._initialize_category_mapping()
     
        self.use_cache = use_cache
        self.image_cache = {}
        self.heatmap_cache = {}
        self.cache_size = cache_size

    
        if validate_files:
            self.validate_dataset_files()

    def _create_group_to_indices(self):
      
        group_to_indices = {}
        for group, attributes in self.attribute_groups.items():
            group_to_indices[group] = [
                self.metadata.columns.get_loc(attr) for attr in attributes
            ]
        return group_to_indices

    def _initialize_category_mapping(self):
     
        # Remap category 
        valid_categories = [c for c in range(1, 51) if c not in [38, 45, 49, 50]]
        logger.info(f"Valid categories after filtering: {valid_categories}")

        self.metadata['original_category_label'] = self.metadata['category_label'].copy()

        self.category_mapping = {
            old_label: new_label for new_label, old_label in enumerate(sorted(valid_categories))
        }
        logger.info(f"Category mapping: {self.category_mapping}")

        # Apply mapping
        self.metadata = self.metadata[self.metadata['category_label'].isin(valid_categories)]
        self.metadata['category_label'] = self.metadata['category_label'].map(self.category_mapping)

       
        assert self.metadata['category_label'].min() == 0, "Category labels should start from 0"
        assert self.metadata['category_label'].max() == 45, "Category labels should range from 0 to 45"

        # Convert category 
        self.metadata['category_type'] = self.metadata['category_type'] - 1

        # Debugging: Verify labels are now 0-indexed
        unique_labels = sorted(self.metadata['category_label'].unique().tolist())
        logger.info(f"Final category labels after remapping: {unique_labels}")
        print(f"Final Category Labels: {unique_labels}")  

        # Create category name mapp
        try:
            self.category_names = dict(zip(
                self.metadata['category_label'].unique(),
                self.metadata['category_name'].unique() if 'category_name' in self.metadata.columns
                else self.metadata['category_label'].unique()
            ))
            logger.info(f"Created category mapping with {len(self.category_names)} categories")
        except Exception as e:
            logger.warning(f"Could not create category name mapping: {str(e)}")
            self.category_names = dict(zip(
                self.metadata['category_label'].unique(),
                self.metadata['category_label'].unique()
            ))

    def __len__(self):
        return len(self.metadata)

    def _get_default_item(self):
        return {
            'image': torch.zeros((3, 224, 224), dtype=torch.float32),  # Blank RGB image
            'heatmap': torch.zeros((1, 224, 224), dtype=torch.float32),  # Blank heatmap
            'attribute_targets': {
                group: torch.zeros(len(indices), dtype=torch.float32)
                for group, indices in self.group_to_indices.items()
            },
            'category_label': torch.tensor(0, dtype=torch.long),  # Default category label
            'category_type': torch.tensor(0, dtype=torch.long),  # Default category type
            'image_name': 'default'  
        }

    def _find_heatmap(self, filename):
        
        for root, _, files in os.walk(self.heatmaps_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def __getitem__(self, idx):
        try:
            row = self.metadata.iloc[idx]

           
            image_name = row['image_name']  
            
            image_path = os.path.join(self.cropped_images_dir, image_name)

           
            base_image_name = os.path.basename(image_name)
            filename_no_ext = os.path.splitext(base_image_name)[0]
            heatmap_filename = f"{filename_no_ext}.npz"
            heatmap_path = self._find_heatmap(heatmap_filename)

            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return self._get_default_item()

            if heatmap_path is None or not os.path.exists(heatmap_path):
                logger.warning(f"Heatmap not found for {image_name}")
                return self._get_default_item()

            image = self._load_image(image_path)
            if image is None:
                logger.warning(f"Could not load image at {image_path}, returning default item.")
                return self._get_default_item()

            
            heatmap = self._load_heatmap(heatmap_path)
            if heatmap is None:
                logger.warning(f"Could not load heatmap at {heatmap_path}, returning default item.")
                return self._get_default_item()

          
            if self.transform:
                seed = torch.initial_seed() % 2**32
                random.seed(seed)
                torch.manual_seed(seed)

               
                if not isinstance(image, torch.Tensor):
                    image = self.transform(image)  #

              
                heatmap = self.transform_heatmap(heatmap)

          
            if not isinstance(image, torch.Tensor):
                logger.warning(f"Image is not a tensor after transform, converting to tensor.")
                image = transforms.ToTensor()(image)

        

            # Debugging: Verify image is a tensor
            if not isinstance(image, torch.Tensor):
                logger.error(f"Image is still not a tensor after conversion, returning default item.")
                return self._get_default_item()

            # Create attribute targets
            attribute_targets = {
                group: torch.tensor(row.iloc[indices].values.astype(np.float32))
                for group, indices in self.group_to_indices.items()
            }

            return {
                'image': image,  # [3, H, W]
                'heatmap': heatmap,  # [1, H, W]
                'attribute_targets': attribute_targets,
                'category_label': torch.tensor(int(row['category_label']), dtype=torch.long),
                'category_type': torch.tensor(row['category_type'], dtype=torch.long),
                'image_name': image_name
            }

        except Exception as e:
            logger.error(f"Error processing item {idx} (image: {row['image_name']}): {str(e)}")
            return self._get_default_item()


    @lru_cache(maxsize=100)
    def _load_image(self, image_path):
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path).convert('RGB')
            image.load()  

            
            image.verify()
            image = Image.open(image_path).convert('RGB')  # Reopen 

            # Ensure size is 224x224 
            if image.size != (224, 224):
                image = image.resize((224, 224))

            # Convert to Tensor immediately (this is crucial)
            image = transforms.ToTensor()(image)
            # logger.info(f"Image type: {type(image)}")  

            return image  # Now it's a tensor of shape

        except Exception as e:
          
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return torch.zeros(3, 224, 224)  # Return a blank tensor of the correct shape


    @lru_cache(maxsize=100)
    def _load_heatmap(self, heatmap_path):
        try:
            if not os.path.exists(heatmap_path):
                raise FileNotFoundError(f"Heatmap not found: {heatmap_path}")

            with np.load(heatmap_path) as data:
                files = data.files
                if not files:
                    logger.error(f"No arrays found in {heatmap_path}")
                    return None  # Return None so we can bail out
                # Fetch first array
                heatmap = data[files[0]].astype(np.float32)

            # Check for NaN or inf
            if np.any(np.isnan(heatmap)) or np.any(np.isinf(heatmap)):
                logger.warning(f"Invalid heatmap values (NaN/inf) at {heatmap_path}, setting to zeros.")
                heatmap = np.zeros_like(heatmap)

            # Normalize [0,1]
            min_val, max_val = heatmap.min(), heatmap.max()
            dynamic_range = max_val - min_val

            if dynamic_range < 1e-3:
                logger.warning(f"Small dynamic range at {heatmap_path}, setting to zeros.")
                heatmap = np.zeros_like(heatmap)
            elif max_val > min_val:
                heatmap = (heatmap - min_val) / (dynamic_range + 1e-7)
                heatmap = np.clip(heatmap, 0, 1)
            else:
                logger.warning(f"Flat heatmap at {heatmap_path}, setting to zeros.")
                heatmap = np.zeros_like(heatmap)

          
            if heatmap.shape != (224, 224):
                heatmap = np.array(Image.fromarray(heatmap).resize((224, 224), Image.BILINEAR))

           
            heatmap = torch.from_numpy(heatmap).unsqueeze(0)
            return heatmap

        except Exception as e:
            logger.error(f"Error loading heatmap {heatmap_path}: {str(e)}")
            return None  

    def default_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def transform_heatmap(self, heatmap): 
        if not isinstance(heatmap, torch.Tensor):
            heatmap = torch.from_numpy(heatmap).float() 
        if heatmap.ndim == 2:  
            heatmap = heatmap.unsqueeze(0)

        heatmap_3ch = heatmap.expand(3, -1, -1)

      
        if self.transform:
            for t in self.transform.transforms:
                if isinstance(t, (transforms.RandomHorizontalFlip,
                                  transforms.RandomRotation,
                                  transforms.RandomAffine,
                                  transforms.RandomResizedCrop)):
                    heatmap_3ch = t(heatmap_3ch)

        # Convert back to single-channel
        heatmap = heatmap_3ch[:1, :, :]
        return heatmap

    def get_category_mapping(self):
        return dict(zip(
            self.metadata['category_label'].unique(),
            self.metadata['category_name'].unique()
            if 'category_name' in self.metadata.columns
            else self.metadata['category_label'].unique()
        ))

    def get_attribute_names(self):

        return self.metadata.columns[4:-2].tolist()

    def validate_dataset_files(self):
        sample_size = min(100, len(self.metadata))
        indices = random.sample(range(len(self.metadata)), sample_size)
        
        missing_images = 0
        missing_heatmaps = 0
        
        for idx in indices:
            row = self.metadata.iloc[idx]
            img_path = os.path.join(self.cropped_images_dir, row['image_name'])

            base_image_name = os.path.basename(row['image_name'])
            filename_no_ext = os.path.splitext(base_image_name)[0]
            test_heatmap_filename = f"{filename_no_ext}.npz"

            found_heatmap = None
            for root, _, files in os.walk(self.heatmaps_dir):
                if test_heatmap_filename in files:
                    found_heatmap = os.path.join(root, test_heatmap_filename)
                    break

            if not os.path.exists(img_path):
                missing_images += 1
            if not found_heatmap or not os.path.exists(found_heatmap):
                missing_heatmaps += 1

        if missing_images > 0 or missing_heatmaps > 0:
            logger.warning(
                f"Found {missing_images} missing images and {missing_heatmaps} missing heatmaps "
                f"in sample of {sample_size}"
            )
        else:
            logger.info(f"Validated {sample_size} random samples, all files exist")

    def get_stats(self):
        """Get dataset statistics."""
        logger.info("Calculating dataset statistics...")

        stats = {
            'num_samples': len(self.metadata),
            'num_attributes': len(self.attribute_groups),
            'num_categories': len(self.metadata['category_label'].unique()),
            'attribute_distributions': {
                group: {
                    attr: self.metadata[attr].value_counts().to_dict() if attr in self.metadata.columns else {}
                    for attr in self.attribute_groups[group]
                } for group in self.attribute_groups
            },
            'category_distribution': self.metadata['category_label'].value_counts().to_dict(),
            'type_distribution': self.metadata['category_type'].value_counts().to_dict()
        }

        return stats


if __name__ == "__main__":
   
    ATTRIBUTE_GROUPS = {
        'color_print': ['red', 'pink', 'floral', 'striped', 'stripe', 'print', 'printed', 'graphic', 'love', 'summer'],
        'neckline': ['v-neck', 'hooded', 'collar', 'sleeveless', 'strapless', 'racerback', 'muscle'],
        'silhouette_fit': ['slim', 'boxy', 'fit', 'skinny', 'shift', 'bodycon', 'maxi', 'mini', 'midi', 'a-line'],
        'style_construction': ['crochet', 'knit', 'woven', 'lace', 'denim', 'cotton', 'chiffon', 'mesh'],
        'details': ['pleated', 'pocket', 'button', 'drawstring', 'trim', 'hem', 'capri', 'sleeve', 'flare', 'skater', 'sheath', 'shirt', 'pencil', 'classic', 'crop']
    }

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    train_metadata_path = "zsehran/collab/metadata_updated_train.csv"
    cropped_images_dir = "data/processed/train_val"
    heatmaps_dir = "data/processed/train_val_heatmaps"

    # Initialize 
    dataset = FashionDataset(
        metadata_path=train_metadata_path,
        cropped_images_dir=cropped_images_dir,
        heatmaps_dir=heatmaps_dir,
        attribute_groups=ATTRIBUTE_GROUPS,
        transform=train_transform,
        use_cache=True,
        cache_size=100,
        validate_files=True
    )

   
    stats = dataset.get_stats()
    print("\nDataset Statistics:")
    print(f"Number of samples: {stats['num_samples']}")
    print(f"Number of attributes: {stats['num_attributes']}")
    print(f"Number of categories: {stats['num_categories']}")
    print(f"Category distribution: {stats['category_distribution']}")
    print(f"Type distribution: {stats['type_distribution']}")
    print("\nAttribute distributions:")
    for group, attr_dist in stats['attribute_distributions'].items():
        print(f"\n{group}:")
        for attr, dist in attr_dist.items():
            print(f"  {attr}: {dist}")

 
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

  
    print("\nTesting DataLoader...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Images shape: {batch['image'].shape}")
        print(f"Heatmaps shape: {batch['heatmap'].shape}")
        print(f"Category labels: {batch['category_label']}")
        print(f"Category types: {batch['category_type']}")
        
        
        for group, targets in batch['attribute_targets'].items():
            print(f"{group} attributes shape: {targets.shape}")
        
        if batch_idx == 2: 
            break

    print("\nDataset script test completed successfully!")
