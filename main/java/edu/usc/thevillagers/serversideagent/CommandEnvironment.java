package edu.usc.thevillagers.serversideagent;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import edu.usc.thevillagers.serversideagent.env.Environment;
import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.text.TextComponentString;
import net.minecraft.world.WorldServer;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.fml.common.FMLCommonHandler;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent.Phase;
import net.minecraftforge.fml.common.gameevent.TickEvent.ServerTickEvent;

public class CommandEnvironment extends CommandBase {
	
	private List<Environment> envs = new ArrayList<>();
	
	public CommandEnvironment() {
		MinecraftForge.EVENT_BUS.register(this);
	}

	@Override
	public String getName() {
		return "e";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/e <env> <agent>";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		if(args.length < 2) {
			sender.sendMessage(new TextComponentString("Incorrect usage ("+getUsage(sender)+")"));
			return;
		}
		WorldServer world = server.worlds[0];
		BlockPos pos;
		if(args.length > 2)  
			pos = parseBlockPos(sender, args, 2, false);
		else
			pos = sender.getPosition();
		try {
			Class<?> clazz = Class.forName("edu.usc.thevillagers.serversideagent.env.Environment"+args[0]);
			Environment env = (Environment) clazz.newInstance();
			env.setSpawnPoint(pos);
			String cmd = "python python/"+args[1]+".py";
			env.init(world, cmd);
			addEnv(env);
		} catch (Exception e) {
			throw new CommandException("Env "+args[0]+" not found ("+e+")", e);
		}
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
	
	@SubscribeEvent
    public void serverTick(ServerTickEvent event) {
		if(FMLCommonHandler.instance().getMinecraftServerInstance().worlds[0].getWorldTime() % 5 == 0) {
			tickEnvs(event.phase);
		}
    }
	
	private void tickEnvs(Phase phase) {
    	Iterator<Environment> iter = envs.iterator();
		while(iter.hasNext()) {
			Environment env = iter.next();
			try {
				switch(phase) {
				case START:
					env.preTick();
					break;
				case END:
					env.postTick();
					break;
				default:
					break;
				}
			} catch(Exception e) {
				env.terminate();
				iter.remove();
				System.err.println("Env "+env.name+" terminated ("+e+")");
			}
		}
    }
	
	public void addEnv(Environment env) {
    	envs.add(env);
    }
}
