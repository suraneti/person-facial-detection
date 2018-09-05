import requests

# Enable debug to print all data
DEBUG = False


class CloudWatch(object):
    """AWS CloudWatch for logging all event
    """

    def __init__(self):
        self.token = 'M3QQe85TAfq9cXKxs5G086jOgHll3rzA'
        self.headers = {'x-token': self.token}

    def logging(self, group, channel, level, message, context):
        """Push Log event to cloudwatch

        Args:
            group (str): Log group's name
            channel (str): Log channel's name
            level (int, str): Level of log
            message (str): Log's message
            context (list, object): Log's data
        """

        payload = { 
                    'group': group,
                    'channel': channel,
                    'level': level,
                    'message': message,
                    'context': context
                    }

        resp = requests.post('http://13.251.54.116:3228/', headers=self.headers, json=payload)

        if DEBUG:
            print(payload)
            print(resp)
