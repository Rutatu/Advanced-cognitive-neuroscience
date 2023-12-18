#!/usr/bin/env python
# coding: utf-8

# # Group analysis on searchlight maps from the FaceWord experiment
# 
# In the FaceWord experiment, participants looked at words and faces while undergoing fMRI.
# P participantwe are presented with a word, which thewould y try to use to predict the emotionfon an upcoming emoji (happy/fearful). Their taswa is to respond wita h button press to indicate which type of emoji was presented![faceWord_img_stim_both.png](attachment:86ba8a07-fdd8-4e5a-beda-7b149364b886.png)
# 
# 21 cognitive science students participated in the experiment over the course of several years (2019-202 as part of courses in cognitive neuroscience at Aarhus University3).
# 
# The experiment consisted of 6 sessions with 60 WordFace trials in ea (30 positive, 30 negative)ch. Each session lasted 10 minutes.
# 
# Each fMRI volume consisted of 45 slices (voxel-size: 2.53x2.53x3mm). Data was acquired with a TR=1s (1H, yielding a total of 600 images per sessionz The raw data can be found in the fMRI-data folder.
# 
# All data was preprocessed using fMRIprep.).
# 
# In this analysis, we have modelled the onsset foeverych positive and negative emojas an individual column in the design matrix, using the Glover HRF. The design matrix includes 22 nuisance variables and a cosine set for highpass filtering ![faceWord_decoding_design_mat.png](attachment:cf5d145b-4c9c-4443-a827-57c99315778c.png)
# 
# O our aim is to see if there is a difference between the t types of emojisz
# 
# To this end, we have conducted a searchlight analysis on 80% of the 360 trials. We now take the accuracy image for each participant and subject them to a 2nd level t-test.
# 
# #### Tasks and questions for assignment 5 are written at the bottom of the notebook.
# 
# 

# <div class="alert alert-success" role="alert">
# 
# # Preamble: Activate environment
# In the first notebook, we installed a python environment. If you haven't don so, please go back to the ```01_setup_virtual_environment.ipynb``` and complete this before proceeding.
# 
# If you closed/stopped the UCloud run between then and now, we will need to reactivate the environment.
# 
# For this we use a bash script with a subset of the lines we used for setting up the environment
# 
# The script called ```activate.sh``` This should be located in your working directory. If not, save the following in a file with this filename.
# 
# ```bash
# . /work/<MY DIRECTORY NUMBER>/virt_env/bin/activate
# python -m ipykernel install --user --name=virt_env
# echo Done! Remember changing the kernel in Jupyter.
# ```
# </div>

# In[1]:


import os
path='/work/student_folders/rutas_folder/notebooks/' 
os.chdir(path)
get_ipython().system('./activate.sh')




# <div class="alert alert-success" role="alert">
#     
# ### Check that we are in the right environment
# 
# </div>

# In[1]:


import sys
print(sys.executable)
#Check that we have something installed.
import nilearn


# In[2]:


# Additional imports
from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pickle
import pandas as pd
import nilearn
import sklearn


# In[3]:


import fpdf
from fpdf import FPDF


# ### Let's load the faceWord searchlight data from one participant

# In[4]:


import pickle

now = datetime.now()
print('Starting cell:',now.strftime("%H:%M:%S"))

f = open('/work/fMRI_data/FaceWordData/WordFace_searchlight_1.pkl', 'rb')
searchlight = pickle.load(f)
f.close()
print(searchlight)

now = datetime.now()
print('Finishing cell:',now.strftime("%H:%M:%S"))


# ### Plotting searchlight from one participant

# In[5]:


from nilearn.plotting import plot_glass_brain, plot_stat_map
from nilearn.image import new_img_like
import numpy as np

from nilearn import plotting
import matplotlib.pyplot as plt
mask_wb_filename='/work/fMRI_data/FaceWordData/BIDS/derivatives/sub-0054/anat/sub-0054_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'

now = datetime.now()
print('Starting cell:',now.strftime("%H:%M:%S"))

searchlight_img = new_img_like(mask_wb_filename, searchlight.scores_)
print(searchlight_img.shape)


plot_glass_brain(searchlight_img, cmap='jet',colorbar=True, threshold=0.6,
                              title=('sub1'),
                              plot_abs=False, display_mode='x')

   
#plt.suptitle('Classification accuracy (unc Acc>0.6)')
plt.show()

plot_stat_map(searchlight_img, cmap='cold_hot',threshold=0.5, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=False,
              title='[image_neg-image_pos] accuracy')
plt.show()

# Saving the objects:
#f = open('/work/MikkelWallentin#6287/WordFace_first_level_models_all_trials_searchlight_all_par.pkl', 'wb')
#pickle.dump([searchlight_all, searchlight_img], f)
#f.close()

now = datetime.now()
print('Finishing cell:',now.strftime("%H:%M:%S"))


# ## Load searchlight maps for each participant
# 

# In[6]:


from nilearn.image import new_img_like, math_img
N_par=21
searchlight_all= np.empty((N_par, 0)).tolist()
searchlight_all=[]
searchlight_demean=[]
conditions_label_all= np.empty((N_par, 0)).tolist()
for i in range(0,N_par):
    text = "Loading file %d\n" % (i+1)
    print(text)
    file_name='/work/fMRI_data/FaceWordData/WordFace_searchlight_'+str(i)+'.pkl'
    f = open(file_name, 'rb')
    searchlight = pickle.load(f)
    searchlight_img = new_img_like(mask_wb_filename, searchlight.scores_)
    searchlight_all.append(searchlight_img)
    searchlight_demean.append(math_img("(img1-0.5)*img2",img1=searchlight_img,img2=mask_wb_filename))
    
    #searchlight_all[i]=searchlight
    f.close()


print(searchlight_all[1])
            
now = datetime.now()
print('Finishing cell:',now.strftime("%H:%M:%S"))


# In[7]:


import matplotlib.pyplot as plt

from nilearn import plotting

#subjects = data["ext_vars"]["participant_id"].tolist()
fig, axes = plt.subplots(nrows=5, ncols=5)
for cidx, map in enumerate(searchlight_all):
    plotting.plot_glass_brain(
        map,
        colorbar=True,
        threshold=0.6,
        vmin=0.5,
        vmax=1,
        title=None,
        axes=axes[int(cidx / 5), int(cidx % 5)],
        plot_abs=False,
        display_mode="x",
    )
plt.show()


# In[8]:


from nilearn.image import new_img_like,  mean_img, concat_imgs

#Concatenate beta maps
searchlight_conc=concat_imgs(searchlight_all)
# Make a mean image
searchlight_mean=mean_img(searchlight_conc)
#Concatenate baseline corrected maps
searchlight_conc_de=concat_imgs(searchlight_demean)
# Make a mean of the baseline-corrected image
searchlight_mean_demean=mean_img(searchlight_conc_de)

plot_glass_brain(searchlight_mean, cmap='jet',colorbar=True, threshold=0.55,
                              title=('Searchlight: Mean accuracy'),
                              plot_abs=False, display_mode='x')

plot_glass_brain(searchlight_mean_demean, cmap='jet',colorbar=True, threshold=0.05,vmin=0,
                              title=('Searchlight: Baseline corrected accuracy'),
                              plot_abs=False, display_mode='x')


plot_stat_map(searchlight_mean, cmap='jet',threshold=0.55, cut_coords=[-20,-10,0,10,20,30,40],
              display_mode='z',  black_bg=False,
              title='Group [image_neg-image_pos] accuracy')
plt.show()
plot_stat_map(searchlight_mean_demean, cmap='jet',threshold=0.05, cut_coords=[-20,-10,0,10,20,30,40],
              display_mode='z',  black_bg=False,
              title='Group [image_neg-image_pos] accuracy')
plt.show()


# In[9]:


from nilearn.glm.second_level import SecondLevelModel
import pandas as pd

second_level_input = searchlight_demean
design_matrix = pd.DataFrame(
    [1] * len(second_level_input),
    columns=["intercept"],
)

second_level_model = SecondLevelModel(smoothing_fwhm=6.0, n_jobs=2)
second_level_model = second_level_model.fit(
    second_level_input,
    design_matrix=design_matrix,
)

z_map = second_level_model.compute_contrast(
    second_level_contrast="intercept",
    output_type="z_score",
)


# In[10]:


from scipy.stats import norm
from nilearn.image import threshold_img
from nilearn.glm import threshold_stats_img

p_val = 0.001
p001_unc = norm.isf(p_val)
print(f"The p<0.001 threshold is {p001_unc:.3g}")

thresholded_map2, threshold2 = threshold_stats_img(
    z_map, alpha=0.05, height_control="fdr"
)
print(f"The FDR=.05 threshold is {threshold2:.3g}")

thresholded_map3, threshold3 = threshold_stats_img(
    z_map, alpha=0.05, height_control="bonferroni"
)
print(f"The p<.05 Bonferroni-corrected threshold is {threshold3:.3g}")

#Remove negative effects from image (they are meaningless)
z_map_disp=threshold_img(
    z_map,
    threshold=0,
    two_sided=False,
)



display = plotting.plot_glass_brain(
    z_map_disp,
    threshold=4,
    vmin=0,
    vmax=12,
    colorbar=True,
    symmetric_cbar=False,
    #display_mode="x",
    plot_abs=False,
    title="pos vs neg face (p<0.001, uncor)",
    figure=plt.figure(figsize=(5, 5)),
)
plotting.show()

display = plotting.plot_glass_brain(
    z_map_disp,
    threshold=threshold2,
    vmin=0,
    vmax=12,
    colorbar=True,
    symmetric_cbar=False,
    #display_mode="x",
    plot_abs=False,
    title="pos vs neg face (p<0.05, FDR)",
    figure=plt.figure(figsize=(5, 5)),
)
plotting.show()

display = plotting.plot_glass_brain(
    z_map_disp,
    threshold=threshold3,
    vmin=0,
    vmax=12,
    colorbar=True,
    symmetric_cbar=False,
    #display_mode="x",
    plot_abs=False,
    title="pos vs neg face (p<0.05, bonf)",
    figure=plt.figure(figsize=(5, 5)),
)
plotting.show()



# ### Find anatomical labels for peak activations
# I will use a function called [atlasreader](https://github.com/miykael/atlasreader).
# 

# In[1]:


import os
os.system('python -m pip install atlasreader')
import atlasreader
from atlasreader import create_output


# In[1]:


create_output(z_map_disp, voxel_thresh=threshold2, cluster_extent=0,direction='both')
#Atlasreader automatically saves results to both .png-files and a csv-file. Look in your working directory.
pd.read_csv('atlasreader_peaks.csv')


# In[11]:


atlasreader_output = pd.read_csv('atlasreader_peaks.csv')


# In[81]:


#excluding column "harvard_oxford" to make the output fit into a standard A4 page
atlasreader_output = atlasreader_output.drop('harvard_oxford', axis=1)


# In[82]:


atlasreader_output


# In[83]:


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# function to draw a table
def _draw_as_table(df, pagesize, fontsize=10):
    alternating_colors = [['white'] * len(df.columns), ['lightgray'] * len(df.columns)] * len(df)
    alternating_colors = alternating_colors[:len(df)]
    fig, ax = plt.subplots(figsize=pagesize)
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values,
                        rowLabels=df.index,
                        colLabels=df.columns,
                        rowColours=['lightblue']*len(df),
                        colColours=['lightblue']*len(df.columns),
                        cellColours=alternating_colors,
                        loc='center')
    the_table.auto_set_font_size(False)  # Turn off auto font size adjustment
    the_table.set_fontsize(fontsize)  # Set font size explicitly
    the_table.auto_set_column_width(col=list(range(len(df.columns))))
    return fig

# function to turn the given data frame into pdf file
def dataframe_to_pdf(df, filename, numpages=(1, 1), pagesize=(11, 8.5), fontsize=7):
    with PdfPages(filename) as pdf:
        nh, nv = numpages
        rows_per_page = len(df) // nh
        cols_per_page = len(df.columns) // nv
        for i in range(0, nh):
            for j in range(0, nv):
                page = df.iloc[(i*rows_per_page):min((i+1)*rows_per_page, len(df)),
                               (j*cols_per_page):min((j+1)*cols_per_page, len(df.columns))]
                fig = _draw_as_table(page, pagesize, fontsize)
                if nh > 1 or nv > 1:
                    # Add a part/page number at bottom-center of page
                    fig.text(0.5, 0.5/pagesize[0],
                             "Part-{}x{}: Page-{}".format(i+1, j+1, i*nv + j + 1),
                             ha='center', fontsize=8)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()


# In[84]:


# turn significant regions´ dataframe into a pdf
dataframe_to_pdf(atlasreader_output, "signif_reg.pdf")


# In[8]:


# Sort clusters by volume in mm value in descending order
sorted_clusters_mm = atlasreader_output.sort_values(by='volume_mm', ascending=False)

# Select the top 10 clusters
top_10_clusters_mm = sorted_clusters_mm.head(10)

# Display the top 10 clusters
print(top_10_clusters_mm)


# In[10]:


# Save the top 10 clusters to a CSV file
top_10_clusters_mm.to_csv('top_10_clusters_mm_FDR.csv')


# In[11]:


pd.read_csv('top_10_clusters_mm_FDR.csv')


# ### Fancy plotting of the results
# 
# 

# In[72]:


from nilearn import plotting
# raw z map
display = plotting.plot_stat_map(z_map_disp, title="Raw z map")


# In[73]:


# z contrast map, Pos vs Neg conditions 
display = plotting.plot_glass_brain(
    z_map_disp,
    threshold=threshold2,
    vmin=0,
    vmax=7.3,
    colorbar=True,
    symmetric_cbar=False,
    cmap = 'RdBu_r',  # Choose a colormap
    plot_abs=False,
    title="Positive vs. Negative Faces (p<0.05, FDR)",
    figure=plt.figure(figsize=(6, 4)),  # Adjust figure size
)
plotting.show()


# ## Tasks and questions for assignment 5
# a) Make a short description of the methods used in the searchlight and group analyses.
# 
# b) Consider the different thresholds. Pick one and try to argue for your choice.
# 
# c) Use the thresholded image and make a table of the significant regions (e.g. using atlasreader - see notebook 14)
# 
# d) Briefly describe the results and discuss the extent to which they are surprising, given the task of observing two emojis and responding with a buttonpress.
# 
# e) Include at least one fancy plot.
# 
# f) Eye-ball the univariate group analysis of the same data conducted in notebook 14. What is the difference, if any? Does the multivariate analysis tell us something we didn't already know+
# 
# g) Smoothing has been applied to the 2nd level analysis. Briefly consider whether this is a good or bad thing, given that multvariate pattern analysis is based on the idea of patterns across voxels (you can also try changing the smoothing level).
# 
# h) The t-test approach for group analyses of searchlight analyses has been widely used, but has also been heavily critized (e.g. see Allefeld 2016; Wang et al. 2020). Briefly consider what we could/should have done, if we had had more time. 
# 
# Individual assignment. Max. 2 pages, excl. figures and references, code as appendix.
# 
# #### References
# Allefeld, C., Görgen, K., & Haynes, J.-D. (2016). Valid population inference for information-based imaging: From the second-level t-test to prevalence inference. Neuroimage, 141, 378-392, https://doi.org/10.1016/j.neuroimage.2016.07.040.
# 
# Wang, Q., Cagna, B., Chaminade, T., & Takerkart, S. (2020). Inter-subject pattern analysis: A straightforward and powerful scheme for group-level MVPA. Neuroimage, 204, 116205, https://doi.org/10.1016/j.neuroimage.2019.116205.
# 
# 

# In[74]:


get_ipython().run_line_magic('notebook', '-e Assignment_05.py')


# In[76]:


from nbconvert import PythonExporter
import nbformat

# Replace 'your_notebook_filename.ipynb' with the actual name of your Jupyter Notebook
notebook_filename = 'Assignment05_Nilearn_faceWord_classification_searchlight_group.ipynb.ipynb'
output_filename = 'Assignment_05.py'

# Read the notebook
with open(notebook_filename, 'r', encoding='utf-8') as notebook_file:
    notebook_content = nbformat.read(notebook_file, as_version=4)

# Create a PythonExporter
python_exporter = PythonExporter()

# Convert the notebook to a Python script
python_script, _ = python_exporter.from_notebook_node(notebook_content)

# Save the Python script to a file
with open(output_filename, 'w', encoding='utf-8') as output_file:
    output_file.write(python_script)

print(f"The Jupyter Notebook '{notebook_filename}' has been saved as a Python script '{output_filename}'.")


