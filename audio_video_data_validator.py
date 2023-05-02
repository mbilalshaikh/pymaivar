from glob import glob
import os 
import shutil

from librosa.core import audio

def get_audio_list():
    audio_path = '/home/muhammadbsheikh/workspace/try/wav/'
    audios = glob(audio_path+'*')
    lst = []
    for item in audios:
        file_name = 'v_'+item.split('.')[0].split('/')[-1]
        lst.append(file_name)    
    return lst

def get_split1_list(train=True):
    if train:
        path = '/home/muhammadbsheikh/workspace/projects/mmaction/mmaction2/data/ucf101/annotations/trainlist01.txt'
    else:
        path = '/home/muhammadbsheikh/workspace/projects/mmaction/mmaction2/data/ucf101/annotations/testlist01.txt'
    
    file_contents = [x.strip().split(' ') for x in open(path)]   
    
    lst = []
    for item in file_contents:
        file_name = item[0].split('/')[1]
        lst.append(file_name)
    return lst 


def get_audio_in_split(audios,split_files):
    
    lst = []
    for item in audios:
        if item in split_files:
            lst.append(item)

    return lst

def get_rep(audio_list,number='1'):
    rep_path = '/home/muhammadbsheikh/workspace/try/ax/'+number
    lst = []
    for item in audio_list:
        file_path = rep_path+'/'+item.replace('v_','')+'.png'
        lst.append(file_path)    
    return lst

def prepare_train_test(train_list,test_list,rep='1'):
    train_path = f'/home/muhammadbsheikh/workspace/try/datax/{rep}/train/'
    test_path = f'/home/muhammadbsheikh/workspace/try/datax/{rep}/test/'


    # copy files to train and test folders while creating directories
    copy_count =0
    for item in train_list:
        # fetch class name
        class_name = item.split('/')[-1].split('_')[0]
        #print(class_name)
        #create dir if it does not exist
        dir_path = train_path+class_name       
        
        if not os.path.exists(dir_path):    
            os.mkdir(dir_path)
            print(item, dir_path)

            shutil.copy(item, dir_path) # copy file
            
        else:
            #print('already exist')
            shutil.copy(item, dir_path) # copy file     
        copy_count +=1    
    print(f'{copy_count} files copied')

    copy_count =0
    for item in test_list:
        # fetch class name
        class_name = item.split('/')[-1].split('_')[0]
        #print(class_name)
        #create dir if it does not exist
        dir_path = test_path+class_name       
        
        if not os.path.exists(dir_path):    
            os.mkdir(dir_path)
            print(item, dir_path)

            shutil.copy(item, dir_path) # copy file
            
        else:
            #print('already exist')
            shutil.copy(item, dir_path) # copy file     
        copy_count +=1
    print(f'{copy_count} files copied')

    pass


if __name__ == "__main__":
    
    audio_files = get_audio_list()
    print(f'Total audio files {len(audio_files)}')
    
    train_s1, test_s1 = get_split1_list(True), get_split1_list(False)
    print(f'Total videos train split 1 files {len(train_s1)}')    
    print(f'Total videos test split 1 files {len(test_s1)}')
    a_train_s1 = get_audio_in_split(audio_files,train_s1)
    print(f'Total audios train split 1 files {len(a_train_s1)}')    
    a_test_s1 = get_audio_in_split(audio_files,test_s1)
    print(f'Total audios test split 1 files {len(a_test_s1)}')
    print(f'Total audios {len(get_audio_in_split(audio_files,test_s1))+len(get_audio_in_split(audio_files,train_s1))}')
    
    print(len(get_rep(a_train_s1,'1')))
    print(len(get_rep(a_test_s1,'1')))
    rep_train,  rep_test = get_rep(a_train_s1), get_rep(a_test_s1)    
    prepare_train_test(rep_train,rep_test,'1')
    
    print(len(get_rep(a_train_s1,'4')))
    print(len(get_rep(a_test_s1,'4')))
    rep_train,  rep_test = get_rep(a_train_s1,'4'), get_rep(a_test_s1,'4')    
    prepare_train_test(rep_train,rep_test,'4')

    
    print(len(get_rep(a_train_s1,'5')))
    print(len(get_rep(a_test_s1,'5')))
    rep_train,  rep_test = get_rep(a_train_s1,'5'), get_rep(a_test_s1,'5')    
    prepare_train_test(rep_train,rep_test,'5')
    
    print(len(get_rep(a_train_s1,'6')))
    print(len(get_rep(a_test_s1,'6')))
    rep_train,  rep_test = get_rep(a_train_s1,'6'), get_rep(a_test_s1,'6')    
    prepare_train_test(rep_train,rep_test,'6')

    print(len(get_rep(a_train_s1,'7')))
    print(len(get_rep(a_test_s1,'7')))
    rep_train,  rep_test = get_rep(a_train_s1,'7'), get_rep(a_test_s1,'7')    
    prepare_train_test(rep_train,rep_test,'7')
    
    print(len(get_rep(a_train_s1,'8')))
    print(len(get_rep(a_test_s1,'8')))
    rep_train,  rep_test = get_rep(a_train_s1,'8'), get_rep(a_test_s1,'8')    
    prepare_train_test(rep_train,rep_test,'8')


    print('success')