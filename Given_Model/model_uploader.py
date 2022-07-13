import boto3

s3 = boto3.resource('s3')
input_bucket = s3.Bucket("mlinput-churn-telecomchurn-852619674999-ap-south-1")


for bucket_object in input_bucket.objects.filter(Prefix='models/'):
    models = bucket_object.key.split('/')[-1]
# mybucket.objects.filter(Prefix='foo/bar')


if len(models) > 0:
    