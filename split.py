import csv
import os

path = os.path.abspath("")

with open(path+'/data/training/train_stances.csv', 'r', encoding='UTF-8') as csvDataFile: 
	csvReader = csv.reader(csvDataFile)

	#sortedlist = sorted(csvReader, key=lambda row: row[2], reverse=False)

	with open(path+'/data/training/training_stances.csv', 'w', newline='', encoding='UTF-8') as csvfile1:
		with open(path+'/data/training/validation_stances.csv', 'w', newline='', encoding='UTF-8') as csvfile2:
			
			filewriter1 = csv.writer(csvfile1)
			filewriter1.writerow(['Headline', 'Body ID', 'Stance'])

			filewriter2 = csv.writer(csvfile2)
			filewriter2.writerow(['Headline', 'Body ID', 'Stance'])

			c1 = 0
			c2 = 0
			c3 = 0
			c4 = 0
			c = 0
			for row in csvReader:
				r = row[2]
				print (row[0])
				print (row[1])
				print (row[2])
				if c != 0:

					if r=="agree":
						if c1%10==0:
							filewriter2.writerow([row[0], row[1], row[2]])
						else:
							filewriter1.writerow([row[0], row[1], row[2]])
						c1 = c1 + 1

					if r=="disagree":
						if c2%10==0:
							filewriter2.writerow([row[0], row[1], row[2]])
						else:
							filewriter1.writerow([row[0], row[1], row[2]])
						c2 = c2 + 1

					if r=="discuss":
						if c3%10==0:
							filewriter2.writerow([row[0], row[1], row[2]])
						else:
							filewriter1.writerow([row[0], row[1], row[2]])
						c3 = c3 + 1

					if r=="unrelated":
						if c4%10==0:
							filewriter2.writerow([row[0], row[1], row[2]])
						else:
							filewriter1.writerow([row[0], row[1], row[2]])
						c4 = c4 + 1

				print (c)
				c = c + 1
				







