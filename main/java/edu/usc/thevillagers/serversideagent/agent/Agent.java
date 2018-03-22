package edu.usc.thevillagers.serversideagent.agent;

import com.mojang.authlib.GameProfile;

import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.server.management.PlayerInteractionManager;
import net.minecraft.util.DamageSource;
import net.minecraft.world.WorldServer;
import net.minecraftforge.fml.common.FMLCommonHandler;

public class Agent extends EntityPlayerMP {
	
	public static final float ROTATION_SPEED = 20;
	
	private AgentBrain brain;
	private int brainCooldown = 1;
	
	public Agent(WorldServer world, GameProfile profile) {
		super(FMLCommonHandler.instance().getMinecraftServerInstance(), world, profile, new PlayerInteractionManager(world));
	}

	@Override
	public void onUpdate() {
		super.onUpdate();
		super.onLivingUpdate();
		useBrain(() -> {
			AgentState state = brain.getState();
			boolean updateBrain = --brainCooldown <= 0;
			if(updateBrain) {
				brainCooldown = state.updatePeriod;
				brain.act();
			}
			this.setPositionAndRotation(posX, posY, posZ, 
					rotationYaw   + state.momentumYaw   * ROTATION_SPEED, 
					rotationPitch + state.momentumPitch * ROTATION_SPEED);
			this.travel(state.strafe, 0, state.forward);
			this.setJumping(state.jump);
			if(updateBrain) {
				state.observe(this);
				brain.observe();
			}
		});
	}
	
	public void setBrain(AgentBrain brain) {
		this.brain = brain;
		brain.init();
	}
	
	@Override
	public void onDeath(DamageSource cause) {
		super.onDeath(cause);
		useBrain(() -> brain.terminate());
		this.world.removeEntity(this);
	}
	
	private void useBrain(Runnable run) {
		if(brain == null) return;
		try {
			run.run();
		} catch(Exception e) {
			System.out.println("Error while using brain");
			e.printStackTrace();
			brain = null;
		}
	}
}
