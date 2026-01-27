from PIL import Image
input_image = Image.open("mona.jpg")
output_image = Image.new("RGB", input_image.size)
width = input_image.width
height = input_image.height
for x in range(width):
    for y in range(height):
        pixel = input_image.getpixel((width-x-1, height-y-1))
        output_image.putpixel((x, y), pixel)
output_image.save("output.png")
