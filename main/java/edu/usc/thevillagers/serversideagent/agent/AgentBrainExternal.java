package edu.usc.thevillagers.serversideagent.agent;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import net.minecraft.block.Block;
import net.minecraft.block.state.IBlockState;
import net.minecraft.entity.Entity;

public class AgentBrainExternal extends AgentBrain {

	private final String command;
	private Process process;
	private DataOutputStream pOut;
	private DataInputStream pIn;
	
	public AgentBrainExternal(String command) {
		this.command = command;
	}

	@Override
	protected AgentState initBrain() {
		try {
			process = Runtime.getRuntime().exec(command);
			pOut = new DataOutputStream(new BufferedOutputStream(process.getOutputStream()));
			pIn = new DataInputStream(new BufferedInputStream(process.getInputStream()));
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
			int updatePeriod = pIn.readInt();
			int obsDist = pIn.readInt();
			System.out.println(command+" started and provided: updatePeriod="+updatePeriod+"; obsDist="+obsDist);
			return new AgentState(updatePeriod, obsDist);
		} catch (Exception e) {
			throw new RuntimeException("Couldn't start agent external brain", e);
		}
	}

	@Override
	public void observe() throws Exception {
		AgentState state = getState();
		pOut.writeFloat((float) state.relativePos.x);
		pOut.writeFloat((float) state.relativePos.y);
		pOut.writeFloat((float) state.relativePos.z);
		pOut.writeFloat(state.yaw);
		pOut.writeFloat(state.pitch);
		for(IBlockState b : state.blocks) {
			pOut.writeInt(Block.getIdFromBlock(b.getBlock()));
		}
		pOut.writeInt(state.entities.size());
		for(Entity e : state.entities) {
			pOut.writeUTF(e.toString());
		}
		pOut.flush();
	}

	@Override
	public void act() throws Exception {
		AgentState state = getState();
		state.forward = pIn.readFloat();
		state.strafe = pIn.readFloat();
		state.momentumYaw = pIn.readFloat();
		state.momentumPitch = pIn.readFloat();
		state.jump = pIn.readBoolean();
		state.crouch = pIn.readBoolean();
		state.attack = pIn.readBoolean();
		state.use = pIn.readBoolean();
	}

	@Override
	public void terminate() {
		process.destroy();
	}
}
