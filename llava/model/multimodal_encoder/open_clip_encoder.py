import torch
import open_clip
from PIL import Image

class OpenCLIPVisionTower(torch.nn.Module):
    def __init__(self, model_name, checkpoint_path, delay_load=False):
        super().__init__()
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.is_loaded = False
        self.preprocess = None
        self.model = None

        if not delay_load:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            print(f'{self.model_name} is already loaded, `load_model` called again, skipping.')
            return
        
        # Load the model and preprocessing from open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name)
        
        # Load pre-trained weights
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt)
        
        self.model.eval()  # Set the model to evaluation mode
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        # Preprocess images
        if isinstance(images, list):
            images = torch.stack([self.preprocess(Image.open(img)) for img in images])
        else:
            images = self.preprocess(Image.open(images))
        
        # Run the model
        image_features = self.model.encode_image(images)
        return image_features

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        return next(self.model.parameters()).device

# The use of methods such as feature selection or accessing specific layers can be added based on the specific outputs of the open_clip model.

# Commenting out the actual instantiation and method calls:
# model_tower = OpenCLIPVisionTower('ViT-L-14', 'path/to/your/checkpoints/RemoteCLIP-ViT-L-14.pt')
# model_tower.load_model()  # if needed to be called explicitly
# features = model_tower.forward('path/to/image.jpg')
