package edu.usc.thevillagers.serversideagent.command;

import java.util.List;

import edu.usc.thevillagers.serversideagent.HighLevelAction;
import edu.usc.thevillagers.serversideagent.recording.WorldRecordRecorder;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventAction;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventBlockChange;
import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.command.WrongUsageException;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.item.ItemStack;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.EntityDamageSource;
import net.minecraft.util.EnumHand;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.text.TextComponentString;
import net.minecraft.world.WorldServer;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.entity.living.LivingAttackEvent;
import net.minecraftforge.event.entity.living.LivingEntityUseItemEvent;
import net.minecraftforge.event.entity.player.PlayerContainerEvent;
import net.minecraftforge.event.entity.player.PlayerInteractEvent;
import net.minecraftforge.event.world.BlockEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent.ServerTickEvent;
import net.minecraftforge.fml.relauncher.Side;

/**
 * Command that controls the recording of a specified area.
 */
public class CommandRecord extends CommandBase {
	
	private WorldRecordRecorder record;
	
	private static enum State{IDLE, RECORDING}
	private State state = State.IDLE;

	public CommandRecord() {
		MinecraftForge.EVENT_BUS.register(this);
	}
	
	@Override
	public String getName() {
		return "rec";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/rec <start|stop> ...";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		if(args.length < 1) throw new WrongUsageException("Missing argument");
		switch(args[0]) {
		case "start":
			if(state != State.IDLE) throw new CommandException("Already "+state);
			if(args.length < 7)  throw new WrongUsageException("Missing argument");
			BlockPos pA = parseBlockPos(sender, args, 1, false);
			BlockPos pB = parseBlockPos(sender, args, 4, false);
			BlockPos from = new BlockPos(	Math.min(pA.getX(), pB.getX()), 
											Math.min(pA.getY(), pB.getY()), 
											Math.min(pA.getZ(), pB.getZ()));
			BlockPos to   = new BlockPos(	Math.max(pA.getX(), pB.getX()), 
											Math.max(pA.getY(), pB.getY()), 
											Math.max(pA.getZ(), pB.getZ()));
			record = new WorldRecordRecorder((WorldServer) sender.getEntityWorld(), from, to);
			record.startRecord();
			state = State.RECORDING;
			sender.sendMessage(new TextComponentString("Recording started"));
			break;
		case "stop":
			if(state != State.RECORDING) throw new CommandException("Actually "+state);
			state = State.IDLE;
			try {
				record.endRecord();
			} catch (Exception e) {
				throw new CommandException("Could not save record", e);
			}
			sender.sendMessage(new TextComponentString("Recording saved"));
			break;
		default:
			throw new WrongUsageException("Unkown argument "+args[0]);
		}
	}
	
	@Override
	public List<String> getTabCompletions(MinecraftServer server, ICommandSender sender, String[] args,
			BlockPos targetPos) {
		if(args.length <= 1) return getListOfStringsMatchingLastWord(args, "start", "stop");
		if(args[0].equals("start")) {
			if(args.length <= 4) return getTabCompletionCoordinate(args, 1, targetPos);
			if(args.length <= 7) return getTabCompletionCoordinate(args, 4, targetPos);
		}
		return super.getTabCompletions(server, sender, args, targetPos);
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
	
	@SubscribeEvent
    public void serverTick(ServerTickEvent event) {
		try {
			switch(event.phase) {
			case START:
				if(state == State.RECORDING) record.startRecordTick();
				break;
			case END:
				if(state == State.RECORDING) record.endRecordTick();
				break;
			}
		} catch (Exception e) {
			System.out.println("Recording failed "+e);
			e.printStackTrace();
			state = State.IDLE;
		}
    }
	
	@SubscribeEvent
	public void blockChange(BlockEvent.NeighborNotifyEvent event) {
		if(state == State.RECORDING && !event.getWorld().isRemote) 
			record.recordEvent(new RecordEventBlockChange(event.getPos(), event.getState()));
	}
	
	@SubscribeEvent
	public void playerInteract(PlayerInteractEvent.EntityInteractSpecific event) {
		if(state == State.RECORDING && event.getSide() == Side.SERVER)
			record.recordEvent(new RecordEventAction(new HighLevelAction(HighLevelAction.Type.USE, HighLevelAction.Phase.INSTANT, 
					event.getEntityPlayer().getEntityId(), event.getHand(), event.getItemStack(), 
					event.getTarget().getEntityId(), null, null, event.getLocalPos())));
	}
	
	@SubscribeEvent
	public void playerInteract(PlayerInteractEvent.RightClickItem event) {
		if(state == State.RECORDING && event.getSide() == Side.SERVER)
			record.recordEvent(new RecordEventAction(new HighLevelAction(HighLevelAction.Type.USE, HighLevelAction.Phase.INSTANT, 
					event.getEntityPlayer().getEntityId(), event.getHand(), event.getItemStack(), 
					-1, null, null, null)));
	}
	
	@SubscribeEvent
	public void playerInteract(PlayerInteractEvent.RightClickBlock event) {
		if(state == State.RECORDING && event.getSide() == Side.SERVER)
			record.recordEvent(new RecordEventAction(new HighLevelAction(HighLevelAction.Type.USE, HighLevelAction.Phase.INSTANT, 
					event.getEntityPlayer().getEntityId(), event.getHand(), event.getItemStack(), 
					-1, event.getPos(), event.getFace(), event.getHitVec())));
	}
	
	@SubscribeEvent
	public void playerInteract(PlayerInteractEvent.LeftClickBlock event) { //TODO how to record when cancelled?
		if(state == State.RECORDING && event.getSide() == Side.SERVER)
			record.recordEvent(new RecordEventAction(new HighLevelAction(HighLevelAction.Type.HIT, HighLevelAction.Phase.START, 
					event.getEntityPlayer().getEntityId(), event.getHand(), event.getItemStack(), 
					-1, event.getPos(), event.getFace(), event.getHitVec())));
	}
	
	@SubscribeEvent
	public void playerInteract(LivingEntityUseItemEvent.Start event) {
		if(state == State.RECORDING && !event.getEntity().getEntityWorld().isRemote)
			record.recordEvent(new RecordEventAction(new HighLevelAction(HighLevelAction.Type.USE, HighLevelAction.Phase.START, 
					event.getEntity().getEntityId(), EnumHand.MAIN_HAND, event.getItem(), //TODO which hand? (compare with itemstack...?)
					-1, null, null, null)));
	}
	
	@SubscribeEvent
	public void playerInteract(LivingEntityUseItemEvent.Stop event) {
		if(state == State.RECORDING && !event.getEntity().getEntityWorld().isRemote)
			record.recordEvent(new RecordEventAction(new HighLevelAction(HighLevelAction.Type.USE, HighLevelAction.Phase.STOP, 
					event.getEntity().getEntityId(), EnumHand.MAIN_HAND, event.getItem(),  //TODO which hand?
					-1, null, null, null)));
	}
	
	@SubscribeEvent
	public void playerContainerInteract(PlayerContainerEvent.Close event) { //TODO no event for moving items in containers?
		if(state == State.RECORDING && !event.getEntity().getEntityWorld().isRemote)
			record.recordEvent(new RecordEventAction(new HighLevelAction(HighLevelAction.Type.USE, HighLevelAction.Phase.STOP, 
					event.getEntityPlayer().getEntityId(), EnumHand.MAIN_HAND, ItemStack.EMPTY,
					-1, null, null, null)));
	}
	
	@SubscribeEvent
	public void playerContainerInteract(LivingAttackEvent event) {
		if(state == State.RECORDING && !event.getEntity().getEntityWorld().isRemote)
			if(event.getSource() instanceof EntityDamageSource) {
				Entity attacker = ((EntityDamageSource) event.getSource()).getImmediateSource();
				if(attacker instanceof EntityPlayer)
					record.recordEvent(new RecordEventAction(new HighLevelAction(HighLevelAction.Type.HIT, HighLevelAction.Phase.INSTANT, 
							attacker.getEntityId(), EnumHand.MAIN_HAND, ((EntityPlayer) attacker).getHeldItem(EnumHand.MAIN_HAND),
							event.getEntity().getEntityId(), null, null, null)));
			}
	}
}
