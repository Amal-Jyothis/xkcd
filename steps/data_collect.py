import logging
import datasets
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from zenml import step

class CollectData:
    '''
    Collect data 
    '''
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def get_data(self):
        logging.info(f'Collecting data from {self.data_path}')
        data = datasets.load_dataset(self.data_path)
        data = data.map(data)
        return data

    def data_extract(self, example: datasets.DatasetDict):   
        try:
            output = {}
            output['text'] = example['title']

            if example["image_url"] is None:
                output["image"] = None
                return output
            
            response = requests.get(example['image_url'])
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                img = Image.open(BytesIO(response.content)).convert("RGB")
                output['image'] = img
                return output
            else:
                logging.error(f"Not a valid image URL:, {example['image_url']}")
                return None
            
        except UnidentifiedImageError:
            logging.error(f"Cannot identify image: {example['image_url']}")
            return None

@step
def collect_data(data_path: str) -> datasets.DatasetDict :
    '''
    Collects data with given folder path or url path of the dataset.
    Arg: 
        data_path: path to the dataset
    return:
        data: collected dataset

    '''
    try:
        data_store = CollectData(data_path)
        data = data_store.get_data()
        return data
    
    except Exception as e:
        logging.error(f'Error while collecting data: {e}')
        raise e