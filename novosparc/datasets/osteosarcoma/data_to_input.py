
import pandas as pd

data = 'data'
batch = '1'

matf = data + "_dge"
geof = data + "_geometry.xlsx"

dge = pd.read_csv(matf, sep=',', index_col=0, header=0)
dge = dge.filter(regex='B' + batch + '_cell*')

geometry = pd.read_excel(geof, 'Batch ' + batch, index_col=0, header=0)  # cells x coordinates
geometry.rename(columns={'x_microns': 'xcoord', 'y_microns': 'ycoord'}, inplace=True)

geometry[['xcoord', 'ycoord']].to_csv('geometry_%s.txt' % batch, sep='\t', index=False)
dge.T.to_csv('dge_%s.txt' % batch, sep='\t', index=False)


