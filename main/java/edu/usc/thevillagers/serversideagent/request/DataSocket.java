package edu.usc.thevillagers.serversideagent.request;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;

public class DataSocket {

	public final Socket socket;
	public final DataInputStream in;
	public final DataOutputStream out;
	
	
	public DataSocket(Socket socket) throws IOException {
		this.socket = socket;
		this.in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
		this.out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
	}
}
