package edu.usc.thevillagers.serversideagent.agent;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.Scanner;

import net.minecraft.block.state.IBlockState;

public class AgentBrainExternal extends AgentBrain {

	private final String command;
	private Process process;
	private PrintStream pOut;
	private Scanner pIn;
	
	public AgentBrainExternal(AgentState state, String command) {
		super(state);
		this.command = command;
	}

	@Override
	public void init() {
		try {
			process = Runtime.getRuntime().exec(command);
			pOut = new PrintStream(process.getOutputStream());
			pIn = new Scanner(process.getInputStream());
			new Thread(() -> {
				BufferedReader err = new BufferedReader(new InputStreamReader(process.getErrorStream()));
				String line;
				try {
					while((line = err.readLine()) != null)
						System.out.println(line);
				} catch (IOException e) {
				}
				System.out.println("Agent terminated");
			}).start();
		} catch (Exception e) {
			throw new RuntimeException("Couldn't start agent external brain", e);
		}
	}

	@Override
	public void observe() {
		pOut.println(state.relativePos.x);
		pOut.println(state.relativePos.y);
		pOut.println(state.relativePos.z);
		pOut.println(state.yaw);
		pOut.println(state.pitch);
		for(IBlockState b : state.blocks) {
			pOut.println(b);
		}
		pOut.flush();
	}

	@Override
	public void act() {
		state.forward = pIn.nextFloat();
		state.strafe = pIn.nextFloat();
		state.momentumYaw = pIn.nextFloat();
		state.momentumPitch = pIn.nextFloat();
		state.jump = pIn.nextBoolean();
		state.crouch = pIn.nextBoolean();
		state.attack = pIn.nextBoolean();
		state.use = pIn.nextBoolean();
	}

	@Override
	public void terminate() {
		process.destroy();
	}
}
