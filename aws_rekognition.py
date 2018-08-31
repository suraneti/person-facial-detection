import boto3
import base64
import cv2

DEBUG = False

class Rekognition(object):
    """ AWS faces rekognition

    Args:
        img (str): Path of image
    """

    def __init__(self, img):
        self.client = boto3.client('rekognition')
        self.s3 = boto3.client('s3')
        self.bucket = 'cto-faces-index'
        self.source_img = img

    def upload_image(self):
        """Upload image to S3
        """

        source_img = self.source_img
        file_name = source_img
        key_name = source_img.replace('.jpg', '')
        self.s3.upload_file(file_name, self.bucket, key_name)

    def face_recognition(self):
        """Face recognition

        Returns:
            response (json): Response data
        """

        response = self.client.search_faces_by_image(
            CollectionId='cto-cnx-team',
            FaceMatchThreshold=95,
            Image={
                'S3Object': {
                    'Bucket': 'cto-faces-index',
                    'Name': self.source_img.replace('.jpg', ''),
                },
            },
            MaxFaces=5,
        )
        
        if DEBUG:
            print(response)
        
        return response
