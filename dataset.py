import warnings
warnings.filterwarnings("ignore")

import os.path as osp, glob, numpy as np, sys, os, glob
import torch
from torch_geometric.data import (Data, Dataset)
import tqdm
import json
from math import pi

import random
random.seed(1001)



class ExtremaRecorder():
    '''
    Records the min and max of a set of values
    Doesn't store the values unless the standard deviation is requested too
    '''
    def __init__(self, do_std=True):
        self.min = 1e6
        self.max = -1e6
        self.mean = 0.
        self.n = 0
        self.do_std = do_std
        if self.do_std: self.values = np.array([])

    def update(self, values):
        self.min = min(self.min, np.min(values))
        self.max = max(self.max, np.max(values))
        n_new = self.n + len(values)
        self.mean = (self.mean * self.n + np.sum(values)) / n_new
        self.n = n_new
        if self.do_std: self.values = np.concatenate((self.values, values))

    def std(self):
        if self.do_std:
            return np.std(self.values)
        else:
            return 0.

    def __repr__(self):
        return (
            '{self.min:+7.3f} to {self.max:+7.3f}, mean={self.mean:+7.3f}{std} ({self.n})'
            .format(
                self=self,
                std='+-{:.3f}'.format(self.std()) if self.do_std else ''
                )
            )

    def hist(self, outfile):
        import matplotlib.pyplot as plt
        figure = plt.figure()
        ax = figure.gca()
        ax.hist(self.values, bins=25)
        plt.savefig(outfile, bbox_inches='tight')


def download_points():
    if not osp.isfile('points.txt'):
        if not osp.isfile('points.tgz'):
            url = 'https://home.fnal.gov/~tjyang/points.tgz'
            print('Downloading', url)
            from urllib.request import urlretrieve
            urlretrieve(url, 'points.tgz')
        print('Extracting points.tgz')
        import tarfile
        tar = tarfile.open('points.tgz', 'r:gz')
        tar.extractall()
        tar.close()
        assert osp.isfile('points.txt')


class DuneDataset(Dataset):
    """PyTorch geometric dataset from processed hit information"""

    @property
    def raw_file_names(self):
        if not hasattr(self, '_raw_file_names'):
            self._raw_file_names = [osp.relpath(f, self.raw_dir) for f in glob.iglob(self.raw_dir + '/*.npz')]
            self._raw_file_names.sort()
        return self._raw_file_names
    
    @property
    def processed_file_names(self):
        if not hasattr(self,'processed_files'):
            self.processed_files = [ f'data_{i}.pt' for i in range(len(self.raw_file_names)) ]
            self.unshuffled_processed_files = self.processed_files[:]
            random.shuffle(self.processed_files)
        return self.processed_files
    
    def __len__(self):
        return len(self.processed_file_names)
    
    def get(self, i):
        # print('Loading', self.processed_dir+'/'+self.processed_files[i])
        data = torch.load(self.processed_dir+'/'+self.processed_files[i])
        return data
    
    def get_npz(self, i):
        '''
        Translates shuffled index back to the original npz file
        '''
        proc = self.processed_file_names[i]
        npz = self.raw_file_names[self.unshuffled_processed_files.index(proc)]
        return np.load(self.raw_dir + '/' + npz)

    def process(self):
        for i, f in tqdm.tqdm(enumerate(self.raw_file_names), total=len(self.raw_file_names)):
            d = np.load(self.raw_dir + '/' + f)
            X = d['X']
            # Normalizations: Gets ~99% of events between 0. and 1.
            X[:,0] = 0.5 + (X[:,0] / 1600.)
            X[:,1] /= 600.
            X[:,2] /= 700.
            X[:,3] /= 250.
            data = Data(
                x = torch.from_numpy(X),
                y = torch.LongTensor(d['y'])
                )
            torch.save(data, self.processed_dir + f'/data_{i}.pt')

    def extrema(self):
        ext_x = ExtremaRecorder()
        ext_y = ExtremaRecorder()
        ext_z = ExtremaRecorder()
        ext_c = ExtremaRecorder()
        for i, f in tqdm.tqdm(enumerate(self.raw_file_names), total=len(self.raw_file_names)):
            d = np.load(self.raw_dir + '/' + f)
            ext_x.update(d['X'][:,0])
            ext_y.update(d['X'][:,1])
            ext_z.update(d['X'][:,2])
            ext_c.update(d['X'][:,3])
        print('Max dims:')
        print('x: ', ext_x)
        print('y: ', ext_y)
        print('z: ', ext_z)
        print('c: ', ext_c)
        ext_x.hist('extrema_x.png')
        ext_y.hist('extrema_y.png')
        ext_z.hist('extrema_z.png')
        ext_c.hist('extrema_c.png')

def make_npzs(data_dir='data', use_max_ntracks=False):
    print('Making individual .npz files...')
    train_outdir = data_dir + '/train/raw/'
    test_outdir = data_dir + '/test/raw/'
    if not osp.isdir(train_outdir): os.makedirs(train_outdir)
    if not osp.isdir(test_outdir): os.makedirs(test_outdir)
    i_event = -1
    current_event = []
    with open("points.txt") as f:
        for line in f:
            if line.startswith('Run '):
                if i_event > -1:
                    current_event = np.array(current_event)
                    X = current_event[:,1:]
                    y = current_event[:,0].flatten()
                    if use_max_ntracks:
                        # Keep only some clusters
                        last_id_kept = y[np.random.randint(2000, 3000)]
                        cutoff_index = np.argmax(y > last_id_kept)
                        X = X[:cutoff_index]
                        y = y[:cutoff_index]
                    # 20% to test, 80% to train
                    outdir = test_outdir if i_event % 5 == 0 else train_outdir
                    np.savez(outdir + f'/{i_event}.npz', X=X, y=y, event_id=event_id)
                    del current_event
                line = line.split()
                event_id = np.array([int(line[1]), int(line[3]), int(line[5])])
                current_event = []
                i_event += 1
                if i_event % 100 == 0: print(i_event)
            else:
                current_event.append([float(e) for e in line.split()])

def main():
    import argparse, shutil
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'action', type=str,
        choices=['reprocess', 'extrema', 'fromscratch', 'onlynpzs'],
        )
    parser.add_argument('-l', '--limited', action='store_true')
    args = parser.parse_args()

    data_dir = 'data_lim' if args.limited else 'data'

    if args.action == 'onlynpzs' or args.action == 'fromscratch':
        download_points()
        if osp.isdir(data_dir): shutil.rmtree(data_dir)
        make_npzs(data_dir=data_dir, use_max_ntracks=args.limited)
        if args.action == 'fromscratch':
            DuneDataset(data_dir+'/train')
            DuneDataset(data_dir+'/test')

    elif args.action == 'reprocess':
        if osp.isdir(data_dir+'/train/processed'): shutil.rmtree(data_dir+'/train/processed')
        DuneDataset(data_dir+'/train')
        if osp.isdir(data_dir+'/test/processed'): shutil.rmtree(data_dir+'/test/processed')
        DuneDataset(data_dir+'/test')

    elif args.action == 'extrema':
        DuneDataset(data_dir+'/train').extrema()


if __name__ == '__main__':
    main()