package edu.usc.thevillagers.serversideagent.agent;

import java.io.IOException;

import edu.usc.thevillagers.serversideagent.env.Environment;
import net.minecraft.entity.player.EntityPlayerMP;

public class Human extends Actor {
	
	public Human(Environment env, EntityPlayerMP human) {
		super(env, human, new AgentActionState());
	}

	@Override
	public void terminate() {
	}

	@Override
	public void observe() throws IOException {
	}

	@Override
	public void observeNoReward() throws IOException {
	}

	@Override
	public void act() throws Exception {
	}
}
