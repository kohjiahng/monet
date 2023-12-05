# Monet Dataset
monet

## Source
Downloaded from [Kaggle](https://www.kaggle.com/competitions/gan-getting-started)

## Data
- 300 monet paintings, in jpg and tfrec format
- 7038 general photos, in jpg and tfrec format

Every image is a 256x256x3 Numpy array with dtype `uint8`

## Format
```yml
{
    monet: {
        image: image
    },
    photo: {
        image: image
    }
}
```



