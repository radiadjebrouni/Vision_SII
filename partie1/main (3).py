import cv2
import numpy as np

def hideData(secret, image):
    print("******START CODING DATA ****")
    if (len(secret)*8> ImgSize(image)): # we want to hide one bit in one pixel (LSB), each letter contains 8 bits
        print ("\n ERROR: THE MSG IS TOO LONG !") 
    else:
        # convert the secret msg to  binary, we use # to define the end of the message
        secret_bin=StrToBin(secret+'#')

        i=0
        max=len(secret_bin)
        
        #modify the LSB for each color of the image pixels
        for pixels in image:
            for p in pixels :
                r=IntoBit(p[0])
                g=IntoBit(p[1])
                b=IntoBit(p[2])

                # Replacing the LSB with one bit of the message
                if i<max:
                    # hide a bit into the red pixel
                    p[0]=int(r[:-1] +secret_bin[i],2)
                    # reconvert the pixel from bin to int
                    i=i+1

                if i<max:
                    # hide a bit into the green pixel
                    p[1]=int(g[:-1]+ secret_bin[i],2)
                    i=i+1

                if i<max:
                    # hide a bit into the blue pixel
                    p[2]=int(b[:-1]+ secret_bin[i],2)
                    i=i+1

                if i >=max:
                    break
    return image


def extractData (image):
    print("\n******START EXTRACTING DATA ****")
    data=""
    ExtractedData=""
    FinalData=""
    for pixels in image:
        for p in pixels :
            r=IntoBit(p[0])
            g=IntoBit(p[1])
            b=IntoBit(p[2])

            # get the LSB of each pixel color
            ExtractedData += r[-1]
            ExtractedData += g[-1]
            ExtractedData += b[-1]
        
    #get characters (separate each 8 bits)
    leng=len(ExtractedData)
    print("\nNumber of pixcels found :"+ str(leng))

    FinalData=[ExtractedData[i:i+8] for i in range(0,leng,8)]
    #for i in range(0,leng,8):
    #     FinalData.append(ExtractedData[i:i+8])

    for bit in FinalData:
        data=data +chr(int(bit,2))
        if(data[-1:])=='#':
            break 
    return data[:-1]


def ImgSize (img):
    return img.shape[0] * img.shape[1] * img.shape[2] 


def StrToBin(string):
    return ''.join([ format(ord(i), "08b") for i in string ])


def IntoBit(pixel):
    return format(pixel,'08b')


def BinToStr(full_binr):
    ascii_str = ""
    for value in full_binr:
        if value!=' ':
            int_value = int(str(value), 2)
            ascii_char = chr(int_value)

            ascii_str+= ascii_char
    return ascii_str


if __name__ == "__main__":
    message=input("\nYour secret DATA? : ")
    #message = "radia! do you like chocolate?"
    image_name=input("\nLink to the image ! : ")
    img = cv2.imread(image_name)

    # Hide the message
    fin=hideData(message, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out="output.jpg"
    cv2.imwrite(out, fin)

    
    print("\nFound DATA :" +extractData(fin))
    # Retrieve the message
    cv2.imshow('Original',img)
    cv2.imshow('Result',fin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    