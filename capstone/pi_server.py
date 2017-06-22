import socket


# Add these lines when pi starts
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("10.12.44.104",4321))
sock.send(b"pi\n")
sock.send(b"bye\n")



# Then open a server socket on port 4322 and accepting connections
# Need to translate from Java Code


# public static void getMessage(BufferedReader in) throws IOException {
#         String message = in.readLine();
#         System.out.println("Message from phone:"+message);
#     }
#
#
#     public static void main(String[] args) throws InterruptedException, IOException {
#
#         ServerSocket serverSocket = new ServerSocket(4322);
#         Socket socket = serverSocket.accept();
#         System.out.println("Waiting for connection");
#         BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
#         PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
#         String message = in.readLine();
#         if (message.equals("hello")){
#             out.println("hi");
#             System.out.println("Connection Established");
#         }
#
#         # Communications etc.

#         getMessage(in);
#         Thread.sleep(4000);
#         out.println("NORTH FOUND");
#         System.out.println("NORTH FOUND");
#
#         getMessage(in);
#         Thread.sleep(4000);
#         out.println("STAR FOUND");
#         System.out.println("STAR FOUND");
#
#     }


