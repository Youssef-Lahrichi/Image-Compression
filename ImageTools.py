from PIL import Image

def convert_to_grayscale(filename):
    img = Image.open(filename).convert('L')
    img.save(filename)
    
convert_to_grayscale("Lighthouse.jpg")
