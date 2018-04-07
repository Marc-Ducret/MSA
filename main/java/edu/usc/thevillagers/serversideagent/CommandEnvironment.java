package edu.usc.thevillagers.serversideagent;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.agent.EntityAgent;
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
	
	private Map<String, Environment> envs = new HashMap<>();
	
	public CommandEnvironment() {
		MinecraftForge.EVENT_BUS.register(this);
	}

	@Override
	public String getName() {
		return "e";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/e <add|remove|agent> <env_id> ...";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		if(args.length < 2) {
			sender.sendMessage(new TextComponentString("Incorrect usage ("+getUsage(sender)+")"));
			return;
		}
		WorldServer world = server.worlds[0];
		String envId = args[1];
		
		switch(args[0]) {
		case "add":
			if(envs.containsKey(envId)) throw new CommandException(envId+" already exists");
			BlockPos origin;
			if(args.length > 3)  
				origin = parseBlockPos(sender, args, 3, false);
			else
				origin = sender.getPosition();
			Environment env = createEnvironment(args[2]);
			env.setOrigin(origin);
			env.init(world);
			envs.put(envId, env);
			break;
			
		case "remove":
			if(!envs.containsKey(envId)) throw new CommandException(envId+" doesn't exist");
			envs.get(envId).terminate();
			envs.remove(envId);
			break;
			
		case "agent":
			if(!envs.containsKey(envId)) throw new CommandException(envId+" doesn't exist");
			for(int i = 2; i < args.length; i ++) {
				String cmd = "python python/agent_"+args[i]+".py";
				env = envs.get(envId);
				Agent a = new Agent(env, new EntityAgent(world, env.name));
				try {
					a.startProcess(cmd);
					a.entity.spawn(env.getOrigin());
					env.newAgent(a);
				} catch (IOException e) {
					throw new CommandException("Cannot start agent ("+e+")");
				}
			}
			break;
			
		default:
			throw new CommandException("Unknown option '"+args[0]+"'");
		}
	}
	
	private Environment createEnvironment(String name) throws CommandException {
		try {
			Class<?> clazz = Class.forName("edu.usc.thevillagers.serversideagent.env.Environment"+name);
			Environment env = (Environment) clazz.newInstance();
			return env;
			
		} catch (Exception e) {
			throw new CommandException("Env "+name+" not found ("+e+")", e);
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
    	Iterator<Environment> iter = envs.values().iterator();
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
}
