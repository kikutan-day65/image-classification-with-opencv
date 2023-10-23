import os
import fitz


def extract_pdf(pdf_path, filename):
    pdf = fitz.open(pdf_path)

    for page_num in range(pdf.page_count):
        page = pdf.load_page(page_num)    
        image = page.get_pixmap()
        image.save(f"resource/{filename[:-4]}-{page_num + 1}.jpg")


def create_dir():

    dir_names = ['diagram', 'text', 'image']
    parent_dir = './'

    # create result directory
    for dir_name in dir_names:
        result_dia = f'result/{dir_name}'
        result_path = os.path.join(parent_dir, result_dia)
        
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        else:
            print(f'{result_path} already exist!')

    cutout_dir = 'cutout-image'
    cutout_path = os.path.join(parent_dir, cutout_dir)

    if not os.path.exists(cutout_path):
        os.makedirs(cutout_path)
    else:
        print(f'{cutout_path} already exist!')

    # create each resource directory
    resource_dir = f'resource'
    resource_path = os.path.join(parent_dir, resource_dir)

    if not os.path.exists(resource_path):
        os.makedirs(resource_path)
    else:
        print(f'{resource_path} already exist!')


def main():

    create_dir()

    dir = 'pdfs_image_classification_task/pdfs'
    for filename in os.listdir(dir):
        pdf_path = os.path.join(dir, filename)
        
        extract_pdf(pdf_path, filename)
        print(f'{filename} extracted successfully!')


if __name__ == "__main__":
    main()