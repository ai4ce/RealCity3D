from pyproj import Proj, transform
from itertools import islice 
import glob
import csv
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon

class GeoProcessing:
    def __init__(self):
        self.input_dir = None
    
    def set_input_directory(self,path):
        self.input_dir = path

    def convert_to_lat_long(self):
        inProj = Proj(init='epsg:5783', preserve_units = True)
        outProj = Proj(init='epsg:4326')
        
        with open('point_cloud.csv', mode='w') as pcf:
            pcw = csv.writer(pcf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            pcw.writerow(["File_Name", "Northing", "Easting"])
            for file in glob.glob(self.input_dir + "*.obj"):
                with open(file, 'r') as fin:
                    line = next(islice(fin, 2, 3))
                    print('line: ',line)
                    words = line.split(' ')
                    northing_local,easting_local = words[-2],words[-1]
                    
                    easting,northing = transform(inProj,outProj,northing_local,easting_local)                    
                    pcw.writerow([file.split("/")[-1], northing, easting])

    def fetch_roi_files(self,x1,y1,x2,y2):
        df = pd.read_csv(self.input_dir)
        df = df[np.logical_and(df["Northing"] > x2, df["Northing"] < x1)]
        df = df[np.logical_and(df["Easting"] > y1,df["Northing"] < y2)]
        df.to_csv('results.csv',index=True)

    def fetch_poly_roi(self,poly):
        df = pd.read_csv(self.input_dir)
        with open(poly, 'r') as reader:
            coords = [(float(point.split(',')[0]),float(point.split(',')[1])) for point in reader.read().split('\n')]
            poly = Polygon(coords)
            print(poly)
        df["Northing"] = df["Northing"].astype(float)
        df["Easting"] = df["Easting"].astype(float)
        df["InPoly"] = False
        print(df.head())
        for index, row in df.iterrows():
            print(Point(row["Northing"], row["Easting"]).within(poly))
            df.at[index,'InPoly'] = Point(row["Northing"], row["Easting"]).within(poly)
        df = df[df['InPoly']]
        df.to_csv('results.csv',index=True)

        

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GeoProcessing Tool')
    parser.add_argument('-mode', '--mode', type=int, default=1, help='0 for search; 1 for extraction from mesh; 2 for polygon based extraction')
    parser.add_argument('-i', '--input-data', type=str, default="./poly_mesh/", help='fill in the input directory/file')
    parser.add_argument('-poly', '--poly', type=str, default="./poly.txt", help='input polygon for searching through the data')
    parser.add_argument('-x1', '--x1', type=float, default=None, help='Top left Northing')
    parser.add_argument('-y1', '--y1', type=float, default=None, help='Top left Easting')
    parser.add_argument('-x2', '--x2', type=float, default=None, help='Bottom right Northing')
    parser.add_argument('-y2', '--y2', type=float, default=None, help='Bottom right Easting')
    args = parser.parse_args()
    
    GeoProc = GeoProcessing()
    GeoProc.set_input_directory(args.input_data)
    if args.mode == 1:
        GeoProc.convert_to_lat_long()
    if args.mode == 0:
        GeoProc.fetch_roi_files(args.x1,args.y1,args.x2,args.y2)
    if args.mode == 2:
        GeoProc.fetch_poly_roi(args.poly)
