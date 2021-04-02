import warnings
warnings.filterwarnings("ignore")

import os.path as osp, glob, numpy as np, sys, os, glob
import torch
from torch_geometric.data import (Data, Dataset)
import tqdm
import json
import uptools
import seutils
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


class DuneDataset(Dataset):
    """PyTorch geometric dataset from processed hit information"""
    
    def download(self):
        pass
    
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
            random.shuffle(self.processed_files)
        return self.processed_files
    
    def __len__(self):
        return len(self.processed_file_names)
    
    def get(self, i):
        # print('Loading', self.processed_dir+'/'+self.processed_files[i])
        data = torch.load(self.processed_dir+'/'+self.processed_files[i])
        return data

    def npz_to_features(self, npz_file):
        d = np.load(npz_file)
        deta = d['eta'] - d['jet_eta']
        deta /= 2.

        dphi = d['phi'] - d['jet_phi']
        dphi %= 2.*pi # Map to 0..2pi range
        dphi[dphi > pi] = dphi[dphi > pi] - 2.*pi # Map >pi to -pi..0 range
        dphi /= 2. # Normalize to approximately -1..1 range
                   # (jet size is 1.5, but some things extend up to 2.)

        # Normalizations kind of guestimated so that 2 standard deviations are within 0..1
        pt = d['pt'] / 30.
        energy = d['energy'] / 100.

        return pt, deta, dphi, energy
    
    def process(self):
        for i, f in tqdm.tqdm(enumerate(self.raw_file_names), total=len(self.raw_file_names)):
            d = np.load(self.raw_dir + '/' + f)
            X = d['X']
            X[:,0] = 0.5 + (X[:,0] / 1600.)
            X[:,1] /= 600.
            X[:,2] /= 700.
            data = Data(
                x = torch.from_numpy(X),
                y = torch.LongTensor(d['y'])
                )
            torch.save(data, self.processed_dir + f'/data_{i}.pt')

    def extrema(self):
        ext_x = ExtremaRecorder()
        ext_y = ExtremaRecorder()
        ext_z = ExtremaRecorder()
        for i, f in tqdm.tqdm(enumerate(self.raw_file_names), total=len(self.raw_file_names)):
            d = np.load(self.raw_dir + '/' + f)
            ext_x.update(d['X'][:,0])
            ext_y.update(d['X'][:,1])
            ext_z.update(d['X'][:,2])
        print('Max dims:')
        print('x: ', ext_x)
        print('y: ', ext_y)
        print('z: ', ext_z)
        ext_x.hist('extrema_x.png')
        ext_y.hist('extrema_y.png')
        ext_z.hist('extrema_z.png')

def make_npzs():
    train_outdir = 'data/train/raw/'
    test_outdir = 'data/test/raw/'
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
                    outdir = test_outdir if i_event % 5 == 0 else train_outdir
                    np.savez(outdir + f'/{i_event}.npz', X=X, y=y)
                    del current_event
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
        choices=['reprocess', 'extrema', 'fromscratch'],
        )
    args = parser.parse_args()

    if args.action == 'fromscratch':
        if osp.isdir('data'): shutil.rmtree('data')
        make_npzs()
        DuneDataset('data/train')
        DuneDataset('data/test')

    elif args.action == 'reprocess':
        if osp.isdir('data/processed'): shutil.rmtree('data/processed')
        DuneDataset('data/train')
        DuneDataset('data/test')

    elif args.action == 'extrema':
        DuneDataset('data/train').extrema()


if __name__ == '__main__':
    main()