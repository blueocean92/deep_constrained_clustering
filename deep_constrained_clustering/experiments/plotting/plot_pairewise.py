
import os
import sys
import time
import random
import re
import json
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from collections import defaultdict


if __name__ == "__main__":
    
    folders = [d for d in os.listdir(".") if os.path.isdir(d) and d != "Legend" and d != "Util"]

    label_dict = {
    	"M": ["0","1","2","3","4","5","6","7","8","9"],
    	"F": ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"],
    	"R": ["corporate/industrial", "government/social", "markets", "economics"]
    }

    for folder in folders:

        print("\nStarting "+folder)

        try:
            latent_files = [f for f in os.listdir(folder) if f.startswith("save")]
            print(latent_files)
        except:
            print("No latent files, Skipping Folder")
            continue

        link_points = []

        try:
            must_links = pd.read_pickle(os.path.join(folder,"mustlinks.pkl"))
            cannot_links = pd.read_pickle(os.path.join(folder,"cannotlinks.pkl"))

            random.seed(1)
            ml_sample = random.sample(range(must_links.shape[0]), must_links.shape[0])
            link_points += must_links.iloc[ml_sample[:20],0].tolist()
            link_points += must_links.iloc[ml_sample[:20],1].tolist()
            random.seed(2)
            cl_sample = random.sample(range(cannot_links.shape[0]), cannot_links.shape[0])
            link_points += cannot_links.iloc[cl_sample[:20],0].tolist()
            link_points += cannot_links.iloc[cl_sample[:20],1].tolist()
        except:
            print("No must link / cannot link, Skipping Folder")
            continue

        try:
            noisy_must_links = pd.read_pickle(os.path.join(folder,"noisymustlinks.pkl"))
            random.seed(3)
            noisy_ml_sample = random.sample(range(noisy_must_links.shape[0]),noisy_must_links.shape[0])
            link_points += noisy_must_links.iloc[noisy_ml_sample[:20],0].tolist()
            link_points += noisy_must_links.iloc[noisy_ml_sample[:20],1].tolist()
        except:
            noisy_must_links = []
            noisy_ml_sample = []

        try:
            noisy_cannot_links = pd.read_pickle(os.path.join(folder,"noisycannotlinks.pkl"))
            random.seed(4)
            noisy_cl_sample = random.sample(range(noisy_cannot_links.shape[0]),noisy_cannot_links.shape[0])
            link_points += noisy_cannot_links.iloc[noisy_cl_sample[:20],0].tolist()
            link_points += noisy_cannot_links.iloc[noisy_cl_sample[:20],1].tolist()
        except:
            noisy_cannot_links = []
            noisy_cl_sample = []

        try:
            with open(os.path.join(folder,"intermediate_results.json"), "r") as fp:
                intermediate_results = json.load(fp)
        except:
            intermediate_results = defaultdict(lambda:defaultdict(lambda:0.0))

        link_points = list(set(link_points))

        # Start Plotting
        for k, file in enumerate(latent_files):

            df = pd.read_pickle(os.path.join(folder,file))
            epoch = re.sub('[^0-9]','', file)

            if folder.startswith("Reuters"):
                latent_full = df.sample(frac=0.75, random_state=7).append(df.iloc[link_points,:])
            else:
                latent_full = df.sample(frac=0.25, random_state=7).append(df.iloc[link_points,:])
            
            latent = latent_full.iloc[:,0:10].copy()

            time_start = time.time()
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=350)
            tsne_results = tsne.fit_transform(latent)
            print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

            latent['tsne-1'] = tsne_results[:,0]
            latent['tsne-2'] = tsne_results[:,1]
            latent["class"] = np.array([label_dict[folder[0]][x] for x in latent_full["y"].tolist()])

            plt.figure(k,figsize=(16,10))
            plt.title("Accuracy: %.2f, NMI: %.2f"%(intermediate_results["acc"][epoch],intermediate_results["nmi"][epoch]))

            sns.scatterplot(
                x="tsne-1", y="tsne-2",
                hue="class",
                palette=sns.color_palette("hls", latent["class"].nunique()),
                data=latent,
                legend="full",
                alpha=0.8,
                s=20
            )


            # plot links
            plot_links = [ {"sample": ml_sample, "link": must_links, "count":10, "style": 'b-', "label": "must link"},
                           {"sample": cl_sample, "link": cannot_links, "count":10, "style": 'r-', "label": "cannot link"},
                           {"sample": noisy_ml_sample, "link": noisy_must_links, "count":10, "style": 'k-', "label": "noisy must link"},
                           {"sample": noisy_cl_sample, "link": noisy_cannot_links, "count":10, "style": 'k:', "label": "noisy cannot link"},
                         ]

            for plot_link in plot_links:
                count = 0
                for i in plot_link["sample"]:
                    if count >= plot_link["count"]:
                        break
                    try:
                        p1 = latent.loc[plot_link["link"].loc[i][0]]
                        p2 = latent.loc[plot_link["link"].loc[i][1]]
                        plt.plot([p1["tsne-1"],p2["tsne-1"]], [p1["tsne-2"],p2["tsne-2"]], plot_link["style"], label=plot_link["label"])
                        count += 1
                    except:
                        pass

            # remove duplicate label for lines
            handles, labels = plt.gca().get_legend_handles_labels()
            newLabels, newHandles = [], []
            for handle, label in zip(handles, labels):
              if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)
            
            #lgd = plt.gca().legend(newHandles, newLabels, loc='center left', bbox_to_anchor=(1, 0.5))
            lgd = plt.gca().legend(newHandles, newLabels, loc='center', bbox_to_anchor=(0.5, -0.10),fancybox=True, ncol=len(newLabels), columnspacing=1.0,handlelength=1.0)

            plt.savefig(os.path.join(folder,folder+"_"+epoch+".png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.clf()
