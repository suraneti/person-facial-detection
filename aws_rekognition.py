from cloudwatch import CloudWatch

import boto3
import cv2

# Enable debug to print all data
DEBUG = False

# Enable cloundwatch will log every event
ENABLE_CLOUDWATCH = True

# Cloudwatch
cloudwatch = CloudWatch()


class Rekognition(object):
    """ AWS faces rekognition

    Args:
        img (str): Path of image
    """

    def __init__(self, img):
        self.client = boto3.client('rekognition')
        self.s3 = boto3.client('s3')
        self.bucket = 'cto-faces-index'
        self.collection_id = 'cto-cnx-team'
        self.threshhold = 95
        self.max_faces = 5
        self.source_img = img

    def upload_image(self):
        """Upload image to S3

        Raises:
            Exception: Raises an exception
        """

        try:
            source_img = self.source_img
            file_name = source_img
            key_name = source_img.replace('.jpg', '')
            self.s3.upload_file(file_name, self.bucket, key_name)
            
            if ENABLE_CLOUDWATCH:
                cloudwatch.logging(
                    group='facial-rekognition',
                    channel='aws-rekognition',
                    level='info',
                    message='Upload image to S3',
                    context={
                        'file_name': file_name,
                        'bucket': self.bucket,
                        'key_name': key_name
                    },
                )
        except Exception as err:
            print(err)

            # Cloudwatch upload image exception log
            if ENABLE_CLOUDWATCH:
                cloudwatch.logging(
                    group='facial-rekognition',
                    channel='aws-rekognition',
                    level='error',
                    message='Upload image',
                    context={
                        'error': err
                    }
                )

    def face_recognition(self):
        """Face recognition

        Returns:
            response (json): Response data

        Raises:
            Exception: Raises an exception
        """
        
        try:
            response = self.client.search_faces_by_image(
                CollectionId=self.collection_id,
                FaceMatchThreshold=self.threshhold,
                Image={
                    'S3Object': {
                        'Bucket': self.bucket,
                        'Name': self.source_img.replace('.jpg', ''),
                    },
                },
                MaxFaces=self.max_faces,
            )
            
            if DEBUG:
                print(response)

            if ENABLE_CLOUDWATCH:
                cloudwatch.logging(
                    group='facial-rekognition',
                    channel='aws-rekognition',
                    level='info',
                    message='Post image to rekognition',
                    context={
                        'CollectionId': self.collection_id,
                        'FaceMatchThreshold': self.threshhold,
                        'Image': {
                            'S3Object': {
                                'Bucket': self.bucket,
                                'Name': self.source_img.replace('.jpg', ''),
                            }
                        },
                        'MaxFaces': self.max_faces,
                        'Response': response
                    },
                )
            
            return response
        
        except Exception as err:
            print(err)
            
            # Cloudwatch facial rekognition exception log
            if ENABLE_CLOUDWATCH:
                cloudwatch.logging(
                    group='facial-rekognition',
                    channel='aws-rekognition',
                    level='error',
                    message='Face rekognition',
                    context={
                        'error': err
                    }
                )
