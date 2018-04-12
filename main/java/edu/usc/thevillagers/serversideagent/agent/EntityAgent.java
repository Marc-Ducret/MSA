package edu.usc.thevillagers.serversideagent.agent;

import java.util.UUID;

import com.mojang.authlib.GameProfile;

import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.network.EnumPacketDirection;
import net.minecraft.network.NetHandlerPlayServer;
import net.minecraft.network.NetworkManager;
import net.minecraft.server.management.PlayerInteractionManager;
import net.minecraft.util.DamageSource;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;
import net.minecraftforge.fml.common.FMLCommonHandler;

public class EntityAgent extends EntityPlayerMP {
	
	public static final float ROTATION_SPEED = 20;
	
	public final AgentActionState actionState;
	
	public EntityAgent(WorldServer world, String name) {
		
		super(FMLCommonHandler.instance().getMinecraftServerInstance(), world, 
				new GameProfile(UUID.randomUUID(), name), new PlayerInteractionManager(world));
		actionState = new AgentActionState();
	}
	
	public void spawn(BlockPos pos) {
		this.setPosition(pos.getX() + .5, pos.getY() + 1, pos.getZ() + .5);
		this.connection = new NetHandlerPlayServer(world.getMinecraftServer(), new NetworkManager(EnumPacketDirection.SERVERBOUND), this);
		FMLCommonHandler.instance().getMinecraftServerInstance().getPlayerList().playerLoggedIn(this);
	}
	
	public void remove() {
		this.world.removeEntity(this);
		FMLCommonHandler.instance().getMinecraftServerInstance().getPlayerList().playerLoggedOut(this);
	}

	@Override
	public void onUpdate() {
		super.onUpdate();
		super.onEntityUpdate();
		super.onLivingUpdate();
		AgentActionState state = actionState;
		state.clamp();
		this.setPositionAndRotation(posX, posY, posZ, 
				rotationYaw   + state.momentumYaw   * ROTATION_SPEED, 
				rotationPitch + state.momentumPitch * ROTATION_SPEED);
		this.travel(state.strafe, 0, state.forward);
		this.setJumping(state.jump);
		this.setSneaking(state.crouch);
	}
	
	@Override
	public void onDeath(DamageSource cause) {
		super.onDeath(cause);
		remove();
	}
}
