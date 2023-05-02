class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/home/muhammadbsheikh/workspace/ucf101/UCF-101'

            # Save preprocess data into output_dir
            output_dir = '/home/muhammadbsheikh/workspace/ucf101/preproc_s2'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/home/muhammadbsheikh/workspace/hmdb51/hmdb'

            output_dir = '/home/muhammadbsheikh/workspace/hmdb51/preprocessed'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/home/muhammadbsheikh/workspace/projects/C3D jfzhang95/c3d-pretrained.pth'