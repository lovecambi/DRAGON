#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
/*	Convert integer in MSB form to LSB
Input:
i - integer in MSB form
Output:
return value - integer i in LSB form	*/
int msb2lsb(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

/*	Read idx3-ubyte image file and store images in the array
Input:
file_name - the directory of the file
Output:
arr - array used to store images	*/
template <typename T>
void read_image(string file_name, vector<vector<T>> &arr)
{
	ifstream file(file_name, ios::binary);	// Open file in binary form
	if (file.is_open())
	{
		// Read the magic number
		int magic_number = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = msb2lsb(magic_number);
		// Read the number of images
		int num_image = 0;
		file.read((char*)&num_image, sizeof(num_image));
		num_image = msb2lsb(num_image);
		// Read the number of rows in each image
		int nr = 0;
		file.read((char*)&nr, sizeof(nr));
		nr = msb2lsb(nr);
		// Read the number of column in each image
		int nc = 0;
		file.read((char*)&nc, sizeof(nc));
		nc = msb2lsb(nc);
		// Initialize the array as a 2-D array, each row corresponds to an image
		// In one row, the image is organized row-wise.
		arr.resize(num_image, vector<T>(nr*nc));
		// Put value of all pixels into the array
		for (int i = 0; i < num_image; i++)
		for (int r = 0; r < nr; r++)
		for (int c = 0; c < nc; c++)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			arr[i][(nr*r) + c] = (T)temp / 255;
		}
	}
	else
		cout << "The file cannot be found" << endl;
}

/*	Read idx3-ubyte label file and store images in the array
Input:
file_name - the directory of the file
Output:
arr - array used to store images	*/
void read_label(string file_name, vector<int> &arr)
{
	ifstream file(file_name, ios::binary);	// Open file in binary form
	if (file.is_open())
	{
		// Read the magic number
		int magic_number = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = msb2lsb(magic_number);
		// Read the number of images
		int num_image = 0;
		file.read((char*)&num_image, sizeof(num_image));
		num_image = msb2lsb(num_image);
		// Initialize the array as an array, each element is the label of corresponding image
		arr.resize(num_image);
		// Put value of all pixels into the array
		for (int i = 0; i < num_image; i++)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			arr[i] = (int)temp;
		}
	}
	else
		cout << "The file cannot be found" << endl;
}
