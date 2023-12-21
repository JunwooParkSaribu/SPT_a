import numpy as np
from XmlModule import xml_to_object, mosaic_to_xml, trxyt_to_xml
from ImageModule import make_image_seqs, make_image


WSL_PATH = '/mnt/c/Users/jwoo/Desktop'
WINDOWS_PATH = 'C:/Users/jwoo/Desktop'

#input_tif = f'{WINDOWS_PATH}/Simulation_Test/Simulation2_50beads_noblink/test2.tif'
#input_trxyt = f'{WINDOWS_PATH}/Simulation_Test/Simulation2_50beads_noblink/test2_slimfast.trxyt'
output_xml = f'{WINDOWS_PATH}/mymethod.xml'
output_img = f'mymethod.png'
output_dir = f'{WINDOWS_PATH}/check/mine'

trxyt_to_xml(f'{WINDOWS_PATH}/receptor_7_low.rpt_tracked.trxyt',
             f'{WINDOWS_PATH}/receptor_7_low.rpt_tracked.xml', cutoff=2)
#mosaic_to_xml(f'{WINDOWS_PATH}/Results.csv', f'{WINDOWS_PATH}/mosaic.xml', cutoff=2)
