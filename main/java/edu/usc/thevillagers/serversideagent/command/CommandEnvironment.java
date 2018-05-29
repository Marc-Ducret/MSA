package edu.usc.thevillagers.serversideagent.command;

import java.util.List;

import edu.usc.thevillagers.serversideagent.agent.Human;
import edu.usc.thevillagers.serversideagent.env.Environment;
import edu.usc.thevillagers.serversideagent.env.EnvironmentManager;
import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.text.TextComponentString;
import net.minecraft.world.WorldServer;

/**
 * This command enables manual environment creation, removal and adding humans to environments.
 */
public class CommandEnvironment extends CommandBase {
	
	private final EnvironmentManager envManager;
	
	public CommandEnvironment(EnvironmentManager envManager) {
		this.envManager = envManager;
	}

	@Override
	public String getName() {
		return "e";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/e <add|remove|player> <env_id> ...";
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
			if(envManager.doesEnvExists(envId)) throw new CommandException(envId+" already exists");
			BlockPos origin;
			if(args.length > 3)  
				origin = parseBlockPos(sender, args, 3, false);
			else
				origin = sender.getPosition();
			Environment env = createEnvironment(args[2]);
			env.setOrigin(origin);
			env.init(world);
			envManager.registerEnv(env, envId);
			break;
			
		case "remove":
			if(!envManager.doesEnvExists(envId)) throw new CommandException(envId+" doesn't exist");
			envManager.getEnv(envId).terminate();
			envManager.removeEnv(envId);
			break;
			
		case "player":
			if(!envManager.doesEnvExists(envId)) throw new CommandException(envId+" doesn't exist");
			env = envManager.getEnv(envId);
			for(int i = 2; i < args.length; i ++) {
				EntityPlayerMP player = getPlayer(server, sender, args[i]);
				if(player == null) throw new CommandException("No such player '"+args[i]+"'");
				env.newActor(new Human(env, player));
			}
			break;
			
		default:
			throw new CommandException("Unknown option '"+args[0]+"'");
		}
	}
	
	private Environment createEnvironment(String envString) throws CommandException {
		try {
			int parStart = envString.indexOf('[');
			String envClassName;
			float[] pars;
			if(parStart >= 0) {
				envClassName = envString.substring(0, parStart);
				String[] parsStrs = envString.substring(parStart+1, envString.indexOf(']')).split(",");
				pars = new float[parsStrs.length];
				for(int i = 0; i < parsStrs.length; i++)
					pars[i] = Float.parseFloat(parsStrs[i]);
			} else {
				envClassName = envString;
				pars = new float[] {};
			}
			Class<?> envClass = envManager.findEnvClass(envClassName);
			Environment env = (Environment) envClass.newInstance();
			env.readPars(pars);
			return env;
			
		} catch (Exception e) {
			throw new CommandException("Env "+envString+" not found ("+e+")", e);
		}
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
	
	@Override
	public List<String> getTabCompletions(MinecraftServer server, ICommandSender sender, String[] args,
			BlockPos targetPos) {
		if(args.length <= 1) return getListOfStringsMatchingLastWord(args, "add", "remove", "player");
		if(args.length <= 2) return getListOfStringsMatchingLastWord(args, envManager.getEnvIds());
		if(args[0].equals("player")) return getListOfStringsMatchingLastWord(args, server.getOnlinePlayerNames());
		return getListOfStringsMatchingLastWord(args);
	}
}
