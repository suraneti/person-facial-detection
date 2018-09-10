import boto3

source_img = 'Nut.jpg'
client = boto3.client('rekognition')

def upload_index_faces(source_img):
    """Upload image to S3 then create faces index
    """

    s3 = boto3.client('s3')
    bucket = 'cto-faces-index'
    file_name = source_img
    key_name = source_img.replace('.jpg', '')
    s3.upload_file(file_name, bucket, key_name)

    response = client.index_faces(
        CollectionId='cto-cnx-team',
        DetectionAttributes=[
        ],
        ExternalImageId=source_img.replace('.jpg', ''),
        Image={
            'S3Object': {
                'Bucket': 'cto-faces-index',
                'Name': source_img.replace('.jpg', ''),
            },
        },
    )

    print(response)

# if __name__ == '__main__':
#     upload_index_faces()