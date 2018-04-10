package edu.usc.thevillagers.serversideagent.test;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Random;

public class IOBench {
	
	static final int N = 10_000_000;
	
	public static void main(String[] args) throws IOException {
		float mb = 4*N / 1_000_000F;
		float pipe = pipeTest();
		System.out.printf("pipe: %.3f s %.3f MB/s \n", pipe, mb / pipe);
		float socket = socketTest();
		System.out.printf("socket: %.3f s %.3f MB/s \n", socket, mb / socket);
		System.out.printf("socket is %.2f times slower\n", socket / pipe);
	}
	
	static Process startProcess(String cmd) throws IOException {
		Process p = Runtime.getRuntime().exec(cmd);
		new Thread(() -> {
			BufferedReader err = new BufferedReader(new InputStreamReader(p.getErrorStream()));
			String line;
			try {
				while((line = err.readLine()) != null)
					System.out.println(line);
			} catch (IOException e) {
			}
			System.out.println("Process "+cmd+" terminated");
		}).start();
		return p;
	}
	
	static float pipeTest() throws IOException {
		Process p = startProcess("python python/bench_io.py pipe");
		return streamTest(p.getInputStream(), p.getOutputStream());
	}
	
	static float socketTest() throws IOException {
		startProcess("python python/bench_io.py socket");
		ServerSocket serv = new ServerSocket(1337);
		Socket sok = serv.accept();
		float res = streamTest(sok.getInputStream(), sok.getOutputStream());
		serv.close();
		return res;
	}
	
	static float streamTest(InputStream sIn, OutputStream sOut) throws IOException {
		DataInputStream in = new DataInputStream(new BufferedInputStream(sIn));
		DataOutputStream out = new DataOutputStream(new BufferedOutputStream(sOut));
		Random rand = new Random();
		int[] buffer = new int[N];
		long start = System.nanoTime();
		
		out.writeInt(N);
		for(int i = 0; i < N; i ++) {
			buffer[i] = rand.nextInt();
			out.writeInt(buffer[i]);
		}
		out.flush();
		for(int i = 0; i < N; i++) {
			if(in.readInt() != buffer[i]) throw new RuntimeException("Integer missmatch! (iter "+i+")");
		}
		
		long end = System.nanoTime();
		return (end - start) / 1_000_000_000F;
	}
}
