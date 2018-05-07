package edu.usc.thevillagers.serversideagent;

import net.minecraft.item.ItemStack;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.util.EnumFacing;
import net.minecraft.util.EnumHand;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;

/**
 * Represents an interaction of an EntityPlayer with an Item, Block or Entity.
 * Can be used to record human actions or to represent an action for an agent to execute. </br>
 * Depending on the action represented, irrelevant fields should be null or -1.
 */
public class HighLevelAction { //TODO inventory actions...
	
	public static enum Type {HIT, USE}
	public static enum Phase {INSTANT, START, STOP}
	
	public final Type actionType;
	public final Phase actionPhase;
	
	public final int actorId;
	public final EnumHand hand;
	public final ItemStack heldItem;
	
	public final int targetEntityId;
	public final BlockPos targetBlockPos;
	public final EnumFacing targetBlockFace;
	public final Vec3d targetHit;
	
	public HighLevelAction(Type actionType, Phase actionPhase, int actorId, EnumHand hand, ItemStack heldItem, 
			int targetEntityId, BlockPos targetBlockPos, EnumFacing targetBlockFace, Vec3d targetHit) {
		this.actionType = actionType;
		this.actionPhase = actionPhase;
		this.actorId = actorId;
		this.hand = hand;
		this.heldItem = heldItem;
		this.targetEntityId = targetEntityId;
		this.targetBlockPos = targetBlockPos;
		this.targetBlockFace = targetBlockFace;
		this.targetHit = targetHit;
	}
	
	public NBTTagCompound toNBT() {
		NBTTagCompound compound = new NBTTagCompound();
		compound.setInteger("ActionType", actionType.ordinal());
		compound.setInteger("ActionPhase", actionPhase.ordinal());
		
		compound.setInteger("ActorId", actorId);
		compound.setInteger("Hand", hand.ordinal());
		compound.setTag("HeldItem", heldItem.serializeNBT());
		
		if(targetEntityId > 0) compound.setInteger("TargetEntityId", targetEntityId);
		if(targetBlockPos != null) {
			compound.setInteger("TargetBlockPosX", targetBlockPos.getX());
			compound.setInteger("TargetBlockPosY", targetBlockPos.getY());
			compound.setInteger("TargetBlockPosZ", targetBlockPos.getZ());
		}
		if(targetBlockFace != null) compound.setInteger("TargetBlockFace", targetBlockFace.ordinal());
		if(targetHit != null) {
			compound.setDouble("TargetHitX", targetHit.x);
			compound.setDouble("TargetHitY", targetHit.y);
			compound.setDouble("TargetHitZ", targetHit.z);
		}
		return compound;
	}
	
	public static HighLevelAction fromNBT(NBTTagCompound compound) {
		Type actionType = Type.values()[compound.getInteger("ActionType")];
		Phase actionPhase = Phase.values()[compound.getInteger("ActionPhase")];
		
		int actorId = compound.getInteger("ActorId");
		EnumHand hand = EnumHand.values()[compound.getInteger("Hand")];
		ItemStack heldItem = ItemStack.EMPTY.copy();
		heldItem.deserializeNBT(compound.getCompoundTag("HeldItem"));
		
		int targetEntityId = compound.hasKey("TargetEntityId") ? compound.getInteger("TargetEntityId") : -1;
		BlockPos targetBlockPos = compound.hasKey("TargetBlockPosX") ? new BlockPos(
						compound.getInteger("TargetBlockPosX"), 
						compound.getInteger("TargetBlockPosY"), 
						compound.getInteger("TargetBlockPosZ")) : null;
		EnumFacing targetBlockFace = compound.hasKey("TargetBlockFace") ? 
				EnumFacing.values()[compound.getInteger("TargetBlockFace")] : null;
		Vec3d targetHit = compound.hasKey("TargetBlockPosX") ? new Vec3d(
						compound.getDouble("TargetHitX"), 
						compound.getDouble("TargetHitY"), 
						compound.getDouble("TargetHitZ")) : null;
				
		return new HighLevelAction(actionType, actionPhase, actorId, hand, heldItem, 
				targetEntityId, targetBlockPos, targetBlockFace, targetHit);
	}
}
