package edu.usc.thevillagers.serversideagent.agent;

import java.util.UUID;

import com.mojang.authlib.GameProfile;

import edu.usc.thevillagers.serversideagent.HighLevelAction;
import edu.usc.thevillagers.serversideagent.HighLevelAction.Phase;
import edu.usc.thevillagers.serversideagent.command.CommandConstant;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.network.NetHandlerPlayServer;
import net.minecraft.network.play.server.SPacketParticles;
import net.minecraft.server.management.PlayerInteractionManager;
import net.minecraft.util.DamageSource;
import net.minecraft.util.EnumParticleTypes;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.MathHelper;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.WorldServer;
import net.minecraftforge.fml.common.FMLCommonHandler;

/**
 * An EntityPlayer controlled by an Agent.
 */
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
		this.connection = new NetHandlerPlayServer(world.getMinecraftServer(), new DummyNetworkManager(), this);
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
		if(state.action != null) executeAction(state.action);
		this.setPositionAndRotation(posX, posY, posZ, 
				rotationYaw + state.momentumYaw   * ROTATION_SPEED * CommandConstant.AGENT_SPEED_FACTOR, 
				MathHelper.clamp(rotationPitch + state.momentumPitch * ROTATION_SPEED * CommandConstant.AGENT_SPEED_FACTOR,
						-CommandConstant.AGENT_PITCH_CLAMP, +CommandConstant.AGENT_PITCH_CLAMP));
		Vec3d move = new Vec3d(state.strafe, 0, state.forward).rotateYaw(-rotationYaw * (float) Math.PI / 180)
				.scale(.15F * CommandConstant.AGENT_SPEED_FACTOR);
		motionX = move.x;
		motionZ = move.z;
		this.travel(0, 0, 0);
		this.setJumping(state.jump);
		this.setSneaking(state.crouch);
	}
	
	private void executeAction(HighLevelAction action) {
		if(action.actionPhase != Phase.STOP) swingArm(action.hand);
		switch(action.actionType) {
		case HIT:
			if(action.targetBlockPos != null) {
				interactionManager.onBlockClicked(action.targetBlockPos, action.targetBlockFace);
				interactionManager.blockRemoving(action.targetBlockPos);
			} else if(action.targetEntityId >= 0) {
				Entity e = world.getEntityByID(action.targetEntityId);
				if(e != null) attackTargetEntityWithCurrentItem(e);
			}
			break;
		case USE:
			switch(action.actionPhase) {
			case INSTANT:
				if(action.targetBlockPos != null) {
					interactionManager.processRightClickBlock(this, world, action.heldItem, action.hand, 
							action.targetBlockPos, action.targetBlockFace, 
							(float) action.targetHit.x, (float) action.targetHit.y, (float) action.targetHit.z);
				} else if(action.targetEntityId >= 0) {
					Entity e = world.getEntityByID(action.targetEntityId);
					if(e != null) {
						if(action.targetHit != null) e.applyPlayerInteraction(this, action.targetHit, action.hand);
						else interactOn(e, action.hand);
					}
				} else {
					interactionManager.processRightClick(this, world, action.heldItem, action.hand);
				}
				break;
				
			case START:
				setActiveHand(action.hand);
				break;
				
			case STOP:
				stopActiveHand();
				break;
			}
			break;
		}
	}
	
	@Override
	public void onDeath(DamageSource cause) { //TODO handle death? (how about players in envs...??)
		setHealth(getMaxHealth());
		FMLCommonHandler.instance().getMinecraftServerInstance().getPlayerList()
			.sendPacketToAllPlayers(new SPacketParticles(EnumParticleTypes.EXPLOSION_NORMAL, true, 
					(float)posX, (float)posY, (float)posZ, .5F, 1, .5F, 0, 16));
	}
}
