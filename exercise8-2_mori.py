import os
import fitz


def extract_pdf(pdf_path, filename):
    pdf = fitz.open(pdf_path)

    for page_num in range(pdf.page_count):
        page = pdf.load_page(page_num)    
        image = page.get_pixmap()

        if os.path.exists(f'resource/{filename[:-4]}'):
            image.save(f"resource/{filename[:-4]}/{page_num + 1}.jpg")
        else:
            print("directory does't exist")


def create_dir(filename):

    dir_names = ['diagram', 'text', 'image']
    parent_dir = './result'

    # create each result directory
    for dir_name in dir_names:
        result_dir = f'{filename[:-4]}/{dir_name}'
        result_path = os.path.join(parent_dir, result_dir)
        
        if not os.path.exists(result_path):
            os.makedirs(result_path)

    # create directory for cutout images
    cutout_dir = 'cutout-image'
    cutout_path = os.path.join(parent_dir, cutout_dir)

    if not os.path.exists(cutout_path):
        os.makedirs(cutout_path)

    # create each resource directory
    parent_dir = f'resource'
    resource_dir = f'{filename[:-4]}'
    resource_path = os.path.join(parent_dir, resource_dir)

    if not os.path.exists(resource_path):
        os.makedirs(resource_path)


def main():

    dir = 'pdfs_image_classification_task/pdfs'
    for filename in os.listdir(dir):
        pdf_path = os.path.join(dir, filename)

        create_dir(filename)
        
        extract_pdf(pdf_path, filename)
        print(f'{filename} page extracted successfully!')


if __name__ == "__main__":
    main()