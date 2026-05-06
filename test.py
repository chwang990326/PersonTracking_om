from torchreid.reid.utils import FeatureExtractor


def main():
    
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='config\osnet_x1_0_imagenet.pth',
        device='cpu'
    )

    image_list = [
        'identity/0/0.png',
        'identity/0/1.png',
        'identity/0/2.png',
        'identity/0/3.png',
        'identity/0/4.png'
    ]

    features = extractor(image_list)

    print(features.shape) # output (5, 512)
    print(features)

if __name__ == '__main__':
    main()