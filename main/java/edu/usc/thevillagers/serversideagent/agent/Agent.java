package edu.usc.thevillagers.serversideagent.agent;

import java.util.UUID;

import com.mojang.authlib.GameProfile;

import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.network.EnumPacketDirection;
import net.minecraft.network.NetHandlerPlayServer;
import net.minecraft.network.NetworkManager;
import net.minecraft.network.play.server.SPacketPlayerListItem;
import net.minecraft.server.management.PlayerInteractionManager;
import net.minecraft.util.DamageSource;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;
import net.minecraftforge.fml.common.FMLCommonHandler;

public class Agent extends EntityPlayerMP {
	
	public static final float ROTATION_SPEED = 20;
	
	private AgentBrain brain;
	private int brainCooldown = 1;
	
	public Agent(WorldServer world, String name) {
		super(FMLCommonHandler.instance().getMinecraftServerInstance(), world, new GameProfile(new UUID(world.rand.nextLong(), world.rand.nextLong()), name), new PlayerInteractionManager(world));
	}
	
	public void spawn(BlockPos pos) {
		this.setPosition(pos.getX() + .5, pos.getY() + 1, pos.getZ() + .5);
		this.connection = new NetHandlerPlayServer(world.getMinecraftServer(), new NetworkManager(EnumPacketDirection.SERVERBOUND), this);
		
		FMLCommonHandler.instance().getMinecraftServerInstance().getPlayerList().sendPacketToAllPlayers(
				new SPacketPlayerListItem(SPacketPlayerListItem.Action.ADD_PLAYER, new EntityPlayerMP[] {this}));
		
		world.spawnEntity(this);
		getServerWorld().getPlayerChunkMap().addPlayer(this);
	}
	
	public void remove() {
		useBrain(() -> brain.terminate());
		this.world.removeEntity(this);
		
		if(!this.world.isRemote) {
			FMLCommonHandler.instance().getMinecraftServerInstance().getPlayerList().sendPacketToAllPlayers(
					new SPacketPlayerListItem(SPacketPlayerListItem.Action.REMOVE_PLAYER, new EntityPlayerMP[] {this}));
			this.getServerWorld().getPlayerChunkMap().removePlayer(this);
		}
	}

	@Override
	public void onUpdate() {
		super.onUpdate();
		super.onEntityUpdate();
		super.onLivingUpdate();
		useBrain(() -> {
			AgentState state = brain.getState();
			boolean updateBrain = --brainCooldown <= 0;
			if(updateBrain) {
				brainCooldown = state.updatePeriod;
				state.clamp();
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
		remove();
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
