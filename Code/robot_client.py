import socket
import sys
from naoqi import ALProxy
from time import sleep
import almath

#Initialize the NAOqi ALProxy for text-to-speech
# sam is the name of our robot

#set robot ip and port here. Chage "" which your robot ip and port_number
robot_ip = ""
robot_port = ""

#set server ip and port here
sever_ip= "localhost"
server_port = "8086"

tts_sam = ALProxy("ALAnimatedSpeech", robot_ip, robot_port)
ms_sam = ALProxy("ALMotion", robot_ip, robot_port)



def look_instruct_back(ms_proxy, angle):
        
        print("inside look and instruct")

        ms_proxy.setStiffnesses("Head", 1.0)

        # Simple command for the HeadYaw joint at 10% max speed
        names = "HeadYaw"
        # angles = 0.0*almath.TO_RAD
        fractionMaxSpeed = 0.2

        angles = angle*almath.TO_RAD
        ms_proxy.setAngles(names,angles,fractionMaxSpeed)

        
def start_client():

  
    client_socket = socket.socket()

    try:
        client_socket.connect((sever_ip, server_port))
        print("Connected to server.")

        while True:
            flag = 0
            # Receive the message from the server
            data = client_socket.recv(4096)
            data = data.decode('utf-8')



            if not data:
                print("No data received. Closing connection.")
                break

            #Decode the received data with utf-8
            print(data)
            try:
                parts = data.split(":", 2)  # Split into at most 3 parts

                if len(parts) == 3:
                    robot_name, next_player_name, robot_narrative = parts
                    robot_name= robot_name.strip()
                    robot_narrative = robot_narrative.strip()
                    tts_sam.say(str(robot_narrative))
                    if next_player_name!=None:
                        turn_phrase = "its your turn "+ next_player_name
                        angle = 45.0
                        look_instruct_back(ms_sam, angle)
                        tts_sam.say(str(turn_phrase))
                        look_instruct_back(ms_sam, 0.0)
                elif len(parts) == 2:
                    _,robot_narrative = parts
                    tts_sam.say(str(robot_narrative))

                else:
                    print("Format unkonwn")
                
                msg = "successful"
                client_socket.send(msg.encode('utf-8'))

            except UnicodeDecodeError as e:
                print("Error decoding message: {}".format(e))

    except socket.error as e:
        print("Socket error: {}".format(e))
    except Exception as e:
        print("An error occurred: {}".format(e))
    finally:
        client_socket.close()
        print("Connection closed.")
    
    

if __name__ == '__main__':
    start_client()